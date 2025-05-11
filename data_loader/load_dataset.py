# datasets/load_dataset.py

from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize(dataset_name="imdb", model_name="bert-base-uncased", split="train", max_length=512):
    """
    Loads and tokenizes a Hugging Face dataset.
    
    Args:
        dataset_name (str): The dataset to load (e.g., 'imdb').
        model_name (str): Name of pretrained model for tokenizer.
        split (str): Dataset split to load (e.g., 'train', 'test').
        max_length (int): Max token length for truncation/padding.
    
    Returns:
        tokenized_dataset (Dataset): Tokenized Hugging Face dataset.
    """
    print(f"Loading dataset: {dataset_name} [{split}]")
    raw_dataset = load_dataset(dataset_name, split=split)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=max_length)

    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    # HuggingFace Trainer requires these columns
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_dataset

# Example usage (remove or comment this if importing elsewhere)
if __name__ == "__main__":
    dataset = load_and_tokenize("imdb", "bert-base-uncased", "train")
    print(dataset[0])
