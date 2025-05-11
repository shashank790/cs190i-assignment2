# train/train_full.py

from data_loader.load_dataset import load_and_tokenize
from models.load_model import load_model
from transformers import TrainingArguments, Trainer, EvalPrediction
import numpy as np
#import evaluate

def compute_metrics(eval_pred: EvalPrediction):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": (predictions == eval_pred.label_ids).mean()}

def main():
    model, tokenizer = load_model(mode="full")
    train_dataset = load_and_tokenize(split="train")
    test_dataset = load_and_tokenize(split="test")

    training_args = TrainingArguments(
        output_dir="./results/full",
        evaluation_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=256,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs/full",
        logging_steps=500,
    )

    #for checking/quick testing
    # training_args = TrainingArguments(
    #     output_dir="./results/full",
    #     per_device_train_batch_size=8,
    #     num_train_epochs=1,
    #     max_steps=100,  # only 100 steps
    #     logging_steps=10,
    #     save_strategy="no",  # skip saving checkpoints
    #     evaluation_strategy="steps",
    #     eval_steps=20,
    # )

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