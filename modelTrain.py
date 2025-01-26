from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Step 1: Load your dataset
dataset = load_dataset("json", data_files={"train": "MayoClinic.json"})

# Split dataset into train and eval (80% train, 20% eval)
train_test_split_data = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split_data["train"]
eval_dataset = train_test_split_data["test"]

# Step 2: Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 3: Tokenize the train and eval datasets
def tokenize_function(examples):
    return tokenizer(
        examples["content"],  # Replace "content" with the correct column name in your dataset
        truncation=True,
        padding="max_length",
        max_length=512,
        return_attention_mask=True,  # Explicitly return the attention mask
    )
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["content"])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["content"])

# Step 4: Prepare data for training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.03,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"  # Disable reporting to W&B or similar
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,  # Use tokenized train dataset
    eval_dataset=tokenized_eval_dataset,    # Use tokenized eval dataset
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 7: Train the model
trainer.train()

# Step 8: Save the fine-tuned model
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")
