from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset('json', data_files={'train': '../data/training_data.jsonl'})['train']

model_name = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = tokenizer(examples['input'], max_length=128, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target'], max_length=128, padding="max_length", truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['input', 'target'])

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=100
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
model.save_pretrained("../models/fine_tuned_model")
tokenizer.save_pretrained("../models/fine_tuned_model")

print("✅ Training complete! Fine-tuned model saved.")