from datasets import Dataset
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification, TrainingArguments, Trainer
from PIL import Image
import json
import torch

# Load annotated dataset
def load_annotated_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return Dataset.from_list(data)

dataset = load_annotated_dataset("dataset.json")

# Debug: Inspect dataset keys
print("Dataset sample:", dataset[0])

# Initialize processor with OCR disabled
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)

# Define label mapping
label_list = ["O", "INVOICE_NUMBER", "DATE", "ORGANIZATION_NAME", "CURRENCY", "TOTAL"]
label_map = {label: idx for idx, label in enumerate(label_list)}

# Debug: Inspect the first sample's labels
print("Label map:", label_map)
print("First sample labels:", dataset[0]['ner_tags'])

# Preprocess data function
def preprocess_data(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    words = examples['words']
    boxes = examples['bbox']
    
    # Debug: Inspect labels before conversion
    print("Original labels:", examples['ner_tags'])
    
    word_labels = [[label_map.get(label, label_map["O"]) for label in doc] for doc in examples['ner_tags']]  # Convert labels to integers
    
    # Debug: Inspect converted labels
    print("Converted labels:", word_labels)

    encoding = processor(images, words, boxes=boxes, word_labels=word_labels, padding="max_length", truncation=True)

    # Debug: Inspect encoding dictionary
    print("Encoding keys:", encoding.keys())
    
    # Add labels for token classification
    encoding["labels"] = encoding["labels"]

    return encoding

# Preprocess dataset
encoded_dataset = dataset.map(preprocess_data, batched=True)

# Initialize the model
model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=len(label_list))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=processor.tokenizer,
)

# Start training
trainer.train()
