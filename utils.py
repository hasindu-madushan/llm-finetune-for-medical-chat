import numpy as np
from datasets import load_dataset, Dataset, DatasetDict


def tokenize_dataset_for_qna(tokenizer, data_df, prompt_template, max_len):
    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda sample: tokenize_for_qna(sample, tokenizer, prompt_template, max_len), batched=False)
    return dataset


def tokenize_for_qna(example, tokenizer, prompt_template, max_len):
    prompt = prompt_template.format(question=example['question'])
    answer = example["answer"] + tokenizer.eos_token
    full_text = prompt + answer
    
    # Tokenize prompt to get its length
    prompt_tokens = tokenizer(
        prompt,
        truncation=False
    )
    
    prompt_len = len(prompt_tokens["input_ids"])

    # Tokenize full sequence once to get the total token count
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len
    )
    
    # Convert to numpy arrays for faster operations
    full_text_len = len(tokenized["input_ids"])
    
    # Tokenize full sequence once
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_attention_mask=True    
    )
    
    # Convert to numpy arrays for faster operations
    input_ids = np.array(tokenized["input_ids"])
    attention_mask = np.array(tokenized["attention_mask"])
    
    # Create labels array and mask prompt portion efficiently
    labels = input_ids.copy()
    # Mask the prompt tokens
    padding_len = max_len - full_text_len
    labels[padding_len:padding_len + prompt_len] = -100
    # print(full_text_len, padding_len, prompt_len)
    
    # Update the tokenized dict with numpy arrays
    tokenized["input_ids"] = input_ids.tolist()
    tokenized["attention_mask"] = attention_mask.tolist()
    tokenized["labels"] = labels.tolist()
    
    return tokenized
