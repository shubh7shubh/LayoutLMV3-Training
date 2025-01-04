import os
import torch
from datasets import Dataset, DatasetDict
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Set constants
MODEL_NAME = "microsoft/layoutlmv3-base"
LABEL_LIST = ["O", "B-Key", "I-Key", "B-Value", "I-Value"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load Data
def load_data():
    """
    Loads your JSON dataset and converts it into Hugging Face's Dataset format.
    Assumes data is in the format where each box has its transcription and label.
    """
    # Sample data
    data = [
        {
            "image": "/path/to/image1.jpg",
            "words": ["Today's", "Date", "is", "01/01/2025"],
            "bbox": [[100, 100, 200, 120], [210, 100, 300, 120], [320, 100, 400, 120], [100, 200, 200, 220]],
            "labels": ["B-Key", "I-Key", "O", "B-Value"]
        },
        {
            "image": "/path/to/image2.jpg",
            "words": ["Member", "Name", "John", "Doe"],
            "bbox": [[100, 100, 200, 120], [210, 100, 300, 120], [320, 100, 400, 120], [420, 100, 520, 120]],
            "labels": ["B-Key", "I-Key", "B-Value", "I-Value"]
        }
    ]

    return Dataset.from_list(data)

# Load dataset
dataset = load_data()

# Step 2: Initialize Processor
processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME)

# Step 3: Preprocess Data
def preprocess_data(examples):
    """
    Prepares data for LayoutLMv3 by tokenizing the words, normalizing bboxes, and associating labels.
    """
    encoding = processor(
        images=examples["image"],
        text=examples["words"],
        boxes=examples["bbox"],
        word_labels=[LABEL2ID[label] for label in examples["labels"]],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return encoding

# Apply preprocessing
encoded_dataset = dataset.map(preprocess_data, batched=True)

# Step 4: Split Dataset
train_test_split = encoded_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Step 5: Initialize Model
model = LayoutLMv3ForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID
).to(device)

# Step 6: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./layoutlmv3-finetuned",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    push_to_hub=False
)

# Step 7: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor,
    data_collator=processor.data_collator
)

# Step 8: Train Model
trainer.train()

# Step 9: Save Fine-Tuned Model
model.save_pretrained("./layoutlmv3-finetuned")
processor.save_pretrained("./layoutlmv3-finetuned")
print("Fine-tuned model saved successfully.")
