# train/train_lora.py

from data_loader.load_dataset import load_and_tokenize
from models.load_model import load_model
from transformers import TrainingArguments, Trainer, EvalPrediction
import numpy as np

def compute_metrics(eval_pred: EvalPrediction):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": (predictions == eval_pred.label_ids).mean()}

def main():
    # You can loop through different ranks here if testing multiple
    model, tokenizer = load_model(mode="lora", lora_rank=8)
    train_dataset = load_and_tokenize(split="train")
    test_dataset = load_and_tokenize(split="test")

    training_args = TrainingArguments(
        output_dir="./results/lora",
        evaluation_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=256,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs/lora",
        logging_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()