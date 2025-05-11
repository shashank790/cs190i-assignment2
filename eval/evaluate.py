# eval/evaluate.py

import argparse
import numpy as np
from data_loader.load_dataset import load_and_tokenize
from models.load_model import load_model
from transformers import Trainer, TrainingArguments, EvalPrediction
import torch

def compute_metrics(eval_pred: EvalPrediction):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": (predictions == eval_pred.label_ids).mean()}

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model.")
    parser.add_argument("--mode", type=str, choices=["full", "head", "lora"], required=True,
                        help="Which fine-tuning mode to evaluate (full, head, lora).")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank (used only if mode is lora).")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned model checkpoint (optional).")

    args = parser.parse_args()

    # Load model + tokenizer
    model, tokenizer = load_model(mode=args.mode, lora_rank=args.lora_rank)

    if args.model_path:
        print(f"Loading model from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))

    # Load test data
    test_dataset = load_and_tokenize(split="test")

    # Dummy training args (Trainer requires them)
    training_args = TrainingArguments(
        output_dir="./eval_output",
        per_device_eval_batch_size=8,
        logging_dir="./logs/eval",
        do_train=False,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    print("Evaluating...")
    results = trainer.evaluate()
    print("Results:", results)

if __name__ == "__main__":
    main()