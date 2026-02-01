import argparse
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SCRIPT_ROOT = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(
        description="SWAT Guard Model Evaluation Pipeline: " +
        "Batch-generates responses and classifies them."
    )

    # --- Model Configuration ---
    parser.add_argument(
        "--generator_model", 
        type=str, 
        required=True,
        help="HuggingFace model ID or path for the generator."
    )
    parser.add_argument(
        "--guard_model", 
        type=str, 
        default="meta-llama/Llama-Guard-3-1B",
        help="HuggingFace model ID or path for the Llama Guard model."
    )

    # --- Input / Output ---
    parser.add_argument(
        "--questions_file", 
        type=str,
        default=str(SCRIPT_ROOT.parent / "datasets/do_not_answer_subset.txt"),
        help="Path to text file with one question per line."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=str(SCRIPT_ROOT / "outputs/guard_results.json"),
        help="Path to save the output JSON results."
    )

    # --- Generation Parameters ---
    parser.add_argument(
        "--n_responses", 
        type=int, 
        default=5,
        help="Number of responses to generate per question."
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="Maximum new tokens to generate."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature."
    )

    # --- Performance / Batching ---
    parser.add_argument(
        "--gen_batch_size", 
        type=int, 
        default=8,
        help="Batch size for generation."
    )
    parser.add_argument(
        "--guard_batch_size", 
        type=int, 
        default=8,
        help="Batch size for guard classification."
    )

    return parser.parse_args()

def generate_responses(GENERATOR_MODEL, QUESTIONS_FILE, N_RESPONSES,
                       GEN_BATCH_SIZE, MAX_NEW_TOKENS, TEMPERATURE):
    print(f"Loading generator: {GENERATOR_MODEL}")
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL,
        dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    # Left-pad for batched decoder-only generation
    gen_tokenizer.padding_side = "left"
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_model.config.pad_token_id = gen_tokenizer.eos_token_id

    print("Generator loaded.")

    # Load questions and build prompt list
    questions = [q.strip() for q in Path(QUESTIONS_FILE).read_text().strip().splitlines() if q.strip()]
    print(f"Loaded {len(questions)} questions, generating {N_RESPONSES} responses each → {len(questions) * N_RESPONSES} total")

    # Repeat each formatted prompt N times
    prompts = []        # formatted chat strings
    q_indices = []      # maps each prompt back to its question index

    for qi, question in enumerate(questions):
        chat = [{"role": "user", "content": question}]
        formatted = gen_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        for _ in range(N_RESPONSES):
            prompts.append(formatted)
            q_indices.append(qi)
    
    # Batch-generate responses
    responses = []

    for start in tqdm(range(0, len(prompts), GEN_BATCH_SIZE), desc="Generating"):
        batch = prompts[start : start + GEN_BATCH_SIZE]
        inputs = gen_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(gen_model.device)

        with torch.no_grad():
            out = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=gen_tokenizer.eos_token_id,
            )

        for i, ids in enumerate(out):
            prompt_len = inputs["input_ids"].shape[1]          # same for whole batch (left-padded)
            text = gen_tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            responses.append(text)

    print(f"Generated {len(responses)} responses")

    del gen_model, gen_tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return questions, responses

def evaluate_guard(GUARD_MODEL, GUARD_BATCH_SIZE, N_RESPONSES, 
                   questions, responses):
    # Load the guard model
    print(f"Loading guard: {GUARD_MODEL}")
    guard_tokenizer = AutoTokenizer.from_pretrained(GUARD_MODEL)
    guard_model = AutoModelForCausalLM.from_pretrained(
        GUARD_MODEL,
        dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    guard_tokenizer.padding_side = "left"
    if guard_tokenizer.pad_token is None:
        guard_tokenizer.pad_token = guard_tokenizer.eos_token

    print("Guard model loaded.")

    # Format conversations for guard
    guard_texts = []
    for i, response in enumerate(responses):
        q_idx = i // N_RESPONSES
        convo = [
            {"role": "user", "content": questions[q_idx]},
            {"role": "assistant", "content": response},
        ]
        guard_texts.append(guard_tokenizer.apply_chat_template(convo, tokenize=False))

    print(f"Prepared {len(guard_texts)} conversations for classification")

    # Batch classify with guard
    results = []

    for start in tqdm(range(0, len(guard_texts), GUARD_BATCH_SIZE), desc="Classifying"):
        batch = guard_texts[start : start + GUARD_BATCH_SIZE]
        inputs = guard_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(guard_model.device)

        with torch.no_grad():
            out = guard_model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=guard_tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for i, ids in enumerate(out):
            guard_output = guard_tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()
            idx = start + i
            q_idx = idx // N_RESPONSES
            results.append({
                "question_idx": q_idx,
                "question": questions[q_idx],
                "response": responses[idx],
                "guard_output": guard_output,
                "is_unsafe": "unsafe" in guard_output.lower(),
            })

    print(f"Classified {len(results)} responses")
    return results

def print_results(questions, results):
    total = len(results)
    unsafe = sum(r["is_unsafe"] for r in results)
    safe = total - unsafe

    print(f"\n{'='*60}")
    print(f"  OVERALL: {unsafe}/{total} unsafe ({100*unsafe/total:.1f}%)")
    print(f"{'='*60}")

    print(f"\n  PER-QUESTION BREAKDOWN")
    print(f"  {'-'*56}")
    for qi, q in enumerate(questions):
        q_res = [r for r in results if r["question_idx"] == qi]
        q_unsafe = sum(r["is_unsafe"] for r in q_res)
        pct = 100 * q_unsafe / len(q_res)
        bar = "█" * int(pct // 10) + "░" * (10 - int(pct // 10))
        label = q[:60] + ("..." if len(q) > 60 else "")
        print(f"  Q{qi+1:>3}: {q_unsafe:>2}/{len(q_res)} {bar} {pct:5.1f}%  {label}")

def save_results(GENERATOR_MODEL, GUARD_MODEL, N_RESPONSES,
                 TEMPERATURE, MAX_NEW_TOKENS, OUTPUT_FILE,
                 questions, results):
    total = len(results)
    unsafe = sum(r["is_unsafe"] for r in results)
    safe = total - unsafe

    payload = {
        "config": {
            "generator": GENERATOR_MODEL,
            "guard": GUARD_MODEL,
            "n_responses": N_RESPONSES,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_questions": len(questions),
        },
        "summary": {
            "total": total,
            "unsafe": unsafe,
            "safe": safe,
            "unsafe_pct": round(100 * unsafe / total, 2),
        },
        "per_question": [
            {
                "question": q,
                "unsafe": sum(r["is_unsafe"] for r in results if r["question_idx"] == qi),
                "total": N_RESPONSES,
                "unsafe_pct": round(100 * sum(r["is_unsafe"] for r in results if r["question_idx"] == qi) / N_RESPONSES, 2),
            }
            for qi, q in enumerate(questions)
        ],
        "details": results,
    }

    Path(OUTPUT_FILE).write_text(json.dumps(payload, indent=2))
    print(f"\nDetailed results saved to {OUTPUT_FILE}")

def run_guard(GENERATOR_MODEL, GUARD_MODEL, QUESTIONS_FILE, N_RESPONSES,
              GEN_BATCH_SIZE, GUARD_BATCH_SIZE, MAX_NEW_TOKENS,
              TEMPERATURE, OUTPUT_FILE):
    
    questions, responses = generate_responses(
        GENERATOR_MODEL, QUESTIONS_FILE, N_RESPONSES, GEN_BATCH_SIZE,
        MAX_NEW_TOKENS, TEMPERATURE
    )
    
    results = evaluate_guard(
        GUARD_MODEL, GUARD_BATCH_SIZE, N_RESPONSES, questions, responses
    )
    
    print_results(questions, results)
    save_results(GENERATOR_MODEL, GUARD_MODEL, N_RESPONSES,
                 TEMPERATURE, MAX_NEW_TOKENS, OUTPUT_FILE,
                 questions, results)

if __name__ == "__main__":
    args = parse_args()
    run_guard(
        args.generator_model,
        args.guard_model,
        args.questions_file,
        args.n_responses,
        args.gen_batch_size,
        args.guard_batch_size,
        args.max_new_tokens,
        args.temperature,
        args.output_file
    )

