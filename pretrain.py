import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch
import wandb

# ==========================================
# Config 
# ==========================================
data_path = "../data/pubmed_baseline/"
train_data_path = data_path + "pubmed_train.csv"
val_data_path = data_path + "pubmed_val.csv"
model_output_dir = "../models/phi_pubmed_pretrained_attempt_1"

max_len = 512

model_id = "microsoft/Phi-3.5-mini-instruct"

# ==========================================
# Hypyerparameters 
# ==========================================
lora_r = 8
lora_alpha = 16
lora_target_modules = ["q_proj", "v_proj", "o_proj"]
batch_size = 32
quantization = None
lora_dropout = 0.05
epochs = 5
learning_rate = 5e-5

# ==========================================
# Dataset 
# ==========================================
def tokenize_dataset(tokenizer, data_df):
    dataset = Dataset.from_pandas(data_df)
    def tokenize(example):
        text = f"<s>#{example['title']}\n{example['abstract']}</s>"
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_attention_mask=True)
    dataset = dataset.map(tokenize, batched=False)
    return dataset


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

val_df = pd.read_csv(val_data_path)
train_df = pd.read_csv(train_data_path)

val_set = tokenize_dataset(tokenizer, val_df)
train_set = tokenize_dataset(tokenizer, train_df.loc[:200_001, :])


wandb.init(
    project="pubmed-pretrain",
    name="attempt_3",
    config={
        "model": model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "batch_size": batch_size,
        "epochs": epochs,
        "quantization": quantization,
        "lora_target_modules": lora_target_modules
    }
)

# ==========================================
# Model 
# ==========================================
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True
# )


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)


model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)


# ==========================================
# Train 
# ==========================================
training_args = TrainingArguments(
    output_dir=model_output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    eval_strategy="epoch",  # ✅ eval at each epoch
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=learning_rate,
    fp16=True,
    report_to="wandb",
    run_name="attempt_3",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

trainer.save_model(model_output_dir + "/final")
tokenizer.save_pretrained(model_output_dir + "/tokenizer")