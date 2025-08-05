import argparse
import json
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import torch

from src.utils import load_beir_datasets, load_cached_data, load_models, save_outputs
from src.defend_module import drift_filtering_question_aware
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.defend_module import PerplexityFilter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run drift filtering on a BEIR dataset"
    )
    parser.add_argument(
        "--eval_dataset", type=str, required=True, help="Dataset name, e.g., nq"
    )
    parser.add_argument(
        "--eval_model_code",
        type=str,
        required=True,
        help="Model code, e.g., contriever",
    )
    parser.add_argument(
        "--score_function",
        type=str,
        choices=["dot", "cos_sim"],
        default="dot",
        help="Scoring function",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Top-k passages to consider"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["drift", "perplexity"],
        required=True,
        help="Filtering method to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split = "test"
    device = "cuda"

    # Load model
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.to(device).eval()

    # Load dataset
    corpus, queries, qrels = load_cached_data(
        f"data_cache/{args.eval_dataset}_{split}.pkl",
        load_beir_datasets,
        args.eval_dataset,
        split,
    )

    log_args = SimpleNamespace(
        log_name=f"extract_val_{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}_{args.method}",
        score_function=args.score_function,
    )

    # Load retrieval results
    result_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}"
    if args.score_function == "cos_sim":
        result_path += "-cos"
    result_path += ".json"

    with open(result_path, "r") as f:
        results = json.load(f)

    # Run filtering based on the selected method
    if args.method == "drift":
        total_removed = 0
        total_queries = 0
        drift_diffs = []
        diff_start_all = []
        diff_end_all = []

        for query_id, topk_dict in tqdm(
            results.items(), desc=f"Evaluating {args.method} impact"
        ):
            question = queries[query_id]
            topk_indices = list(topk_dict.keys())[: args.top_k]
            original_docs = [corpus[doc_id]["text"] for doc_id in topk_indices]

            try:
                filtered_docs, diff_start, diff_end = drift_filtering_question_aware(
                    args=log_args,
                    tokenizer=tokenizer,
                    model=model,
                    get_emb=get_emb,
                    question=question,
                    contents=original_docs,
                    score=args.score_function,
                )
            except Exception as e:
                print(f"Skipping {query_id} due to error: {e}")
                continue

            removed = len(set(original_docs) - set(filtered_docs))
            drift_diffs.append(removed)
            diff_start_all = np.concatenate((diff_start_all, diff_start))
            diff_end_all = np.concatenate((diff_end_all, diff_end))
            total_removed += removed
            total_queries += 1

        # Report
        avg_removed = total_removed / total_queries if total_queries > 0 else 0
        print("\n[Contriever Drift Filter Summary]")
        print(f"Total queries processed: {total_queries}")
        print(
            f"Average documents removed by drift: {avg_removed:.2f} out of top-{args.top_k}"
        )

        save_outputs(
            diff_start_all,
            log_args.log_name,
            f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-diff_start_all",
        )
        save_outputs(
            diff_end_all,
            log_args.log_name,
            f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-diff_end_all",
        )
    elif args.method == "perplexity":
        # You may want to add these as arguments
        gen_model_name = "meta-llama/Llama-3.1-8B-Instruct"

        print(f"Loading generation model: {gen_model_name}")
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

        gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name).to(device)
        gen_model.eval()

        all_nlls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        i = 0
        for query_id, topk_dict in tqdm(
            results.items(), desc="Calculating NLLs for perplexity threshold"
        ):
            i += 1
            if i % 500 == 0:
                break
            topk_indices = list(topk_dict.keys())[: args.top_k]
            original_docs = [corpus[doc_id]["text"] for doc_id in topk_indices]

            try:
                # Get NLLs for each doc (do not filter, just collect NLLs)
                input_ids = gen_tokenizer(
                    original_docs, return_tensors="pt", padding=True, truncation=True
                ).input_ids.to(device)
                with torch.no_grad():
                    logits = gen_model(input_ids, labels=input_ids).logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    # Calculate loss for each token, then mean for each sequence
                    losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    losses = losses.view(shift_labels.size())
                    nlls = losses.mean(dim=1).cpu().numpy()
                    all_nlls.extend(nlls)
            except Exception as e:
                print(f"Skipping {query_id} due to error: {e}")
                continue

        all_nlls = np.array(all_nlls)
        print(f"\nCollected {len(all_nlls)} NLL values.")
        percentiles = [95, 90, 85]
        all_threshs = np.percentile(all_nlls, percentiles)
        for pct, thresh in zip(percentiles, all_threshs):
            print(f"{pct}th percentile NLL: {thresh:.4f}")

        # Optionally save all NLLs for further analysis
        save_outputs(
            all_threshs,
            log_args.log_name,
            f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-nlls_all",
        )
    else:
        raise ValueError("Unknown method")


if __name__ == "__main__":
    main()
