from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('/Users/almonsubba/Desktop/pdf_app/dataset.json')

# Initialize the tokenizer and model
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=len(label_list))

# Preprocess the data
def preprocess_data(examples):
    encoding = tokenizer(examples['words'], boxes=examples['bbox'], truncation=True, padding=True, return_tensors="pt")
    encoding['labels'] = [label_map[label] for label in examples['ner_tags']]
    return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True)

# Training code (simplified)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
)

trainer.train()
