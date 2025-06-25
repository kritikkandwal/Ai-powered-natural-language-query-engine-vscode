from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import torch

# 1. Load dataset
dataset = load_dataset('json', 
                      data_files={
                          'train': '../data/training_data.jsonl'
                      })

# 2. Load pretrained model and tokenizer
model_name = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 3. Preprocess data
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['input'],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['target'],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. Training configuration
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    logging_steps=100,
    save_steps=1000,
)

# 5. Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# 6. Start training
trainer.train()

# 7. Save fine-tuned model
model.save_pretrained("../models/fine_tuned_model")
tokenizer.save_pretrained("../models/fine_tuned_model")