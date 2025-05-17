import numpy as np
from datasets import load_dataset, Dataset, DatasetDict


def tokenize_dataset_for_qna(tokenizer, data_df, prompt_template, max_len):
    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda sample: tokenize_for_qna(sample, tokenizer, prompt_template, max_len), batched=False)
    return dataset


def tokenize_dataset_for_domain_bound_qna(tokenizer, data_df, prompt_template, max_len):
    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda sample: tokenize_for_domain_bound_qna(sample, tokenizer, prompt_template, max_len), batched=False)
    return dataset


def tokenize_for_domain_bound_qna(example, tokenizer, prompt_template, max_len):
    example["answer"] = f"<{example['class']}>{example['answer']}"
    return tokenize_for_qna(example, tokenizer, prompt_template, max_len)
    

def tokenize_for_qna(example, tokenizer, prompt_template, max_len):
    prompt = prompt_template.format(question=example['question'])
    answer = example["answer"] + tokenizer.eos_token
    full_text = prompt + answer

    # Tokenize answer to get the length of answer tokens
    answer_tokens = tokenizer(
        answer,
        truncation=True,
        max_length=max_len
    )
    
    answer_len = len(answer_tokens["input_ids"])
    
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
    labels[:-answer_len] = -100
    
    # Update the tokenized dict with numpy arrays
    tokenized["input_ids"] = input_ids.tolist()
    tokenized["attention_mask"] = attention_mask.tolist()
    tokenized["labels"] = labels.tolist()
    
    return tokenized
