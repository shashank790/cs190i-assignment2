# models/load_model.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn

def load_model(model_name="bert-base-uncased", num_labels=2, mode="full", lora_rank=8):
    """
    Loads and configures a model for fine-tuning.

    Args:
        model_name (str): Hugging Face model name (e.g., 'bert-base-uncased').
        num_labels (int): Number of output labels for classification.
        mode (str): One of ['full', 'head', 'lora'].
        lora_rank (int): LoRA rank if mode='lora'.

    Returns:
        model: The configured model ready for training.
        tokenizer: The associated tokenizer.
    """
    print(f"Loading model: {model_name} with mode: {mode}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if mode == "head":
        print("Freezing all base model parameters (head-only tuning)...")
        for name, param in model.named_parameters():
            if "classifier" not in name and "score" not in name:  # Model head names vary; 'classifier' works for BERT
                param.requires_grad = False

    elif mode == "lora":
        print(f"Applying LoRA with rank {lora_rank}...")
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=16,
            task_type=TaskType.SEQ_CLS,
            lora_dropout=0.1,
            bias="none"
        )
        model = get_peft_model(model, peft_config)

    # 'full' mode: nothing to freeze — all params are tunable by default

    return model, tokenizer