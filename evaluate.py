import argparse
from pathlib import Path

from temporal_embeddings.evaluation.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--model_name", type=str, help="Name of the model to evaluate")
    parser.add_argument("--model_path", type=str, help="Path to the model .pth file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length for evaluation")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark to use for evaluation")
    parser.add_argument("--eval_id", type=int, required=True, help="Evaluation ID to identify the experiment")
    parser.add_argument("--top_k", type=int, default=1, help="Value of k for top-k accuracy")
    parser.add_argument("--metric", type=str, default="top", choices=["all", "top", "mrr", "ndcg", "precision", "recall", "f1"], help="Metric to use for evaluation")
    parser.add_argument("--skip", action="store_true", help="Skip model evaluation")
    parser.add_argument("--external_model_name", type=str, default="", help="Name of external model to integrate with temporal embeddings")
    parser.add_argument("--use_ranking", action="store_true", help="Use ranking for evaluation")
    args = parser.parse_args()

    evaluate_model(args.model_name, args.external_model_name, Path(args.model_path), args.batch_size, args.max_seq_len, args.benchmark, args.eval_id, args.top_k, args.metric, args.skip, args.use_ranking)

if __name__ == "__main__":
    main()