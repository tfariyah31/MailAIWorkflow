from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os
import logging

# Handle AdamW import for different transformers versions
try:
    from transformers import AdamW  # Older versions
except ImportError:
    from torch.optim import AdamW  # Newer versions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force CPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)  # Optimize CPU threads

class Classifier:
    RECRUITER_KEYWORDS = {
        "test automation", "automation engineer", "software qa", "qa engineer",
        "lead qa", "sdet", "quality engineer", "qa automation", "test engineer", 
        "jd", "job desc", "need for sdet", "looking for qa", "job", "hiring", "urgent Need:SDET"
    }

    NON_RECRUITER_KEYWORDS = {
        "newsletter", "meeting", "invoice", "security alert", "webinar",
        "notification", "receipt", "subscription", "follow", "invitation", "alert", "reminder", "message"
    }

    def __init__(self, model_name="distilbert-base-uncased", fine_tune_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if fine_tune_path and os.path.exists(fine_tune_path):
            self.model = DistilBertForSequenceClassification.from_pretrained(fine_tune_path)
            logger.info(f"Loaded fine-tuned model from {fine_tune_path}")
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2
            )
            logger.info(f"Initialized new {model_name} model")
        
        self.model.to(device)
        self.model.eval()

    def quick_filter(self, subject: str) -> str | None:
        subject_lower = subject.lower()
        if any(kw in subject_lower for kw in self.RECRUITER_KEYWORDS):
            return "recruiter"
        if any(kw in subject_lower for kw in self.NON_RECRUITER_KEYWORDS):
            return "non-recruiter"
        return None

    def preprocess_data(self, texts, labels):
        """Tokenize data for training"""
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128)
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'label': labels
        })

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, output_dir="./model"):
        """Robust training method that works with all versions"""
        train_dataset = self.preprocess_data(train_texts, train_labels)
        
        if val_texts and val_labels:
            val_dataset = self.preprocess_data(val_texts, val_labels)
        else:
            val_dataset = None

        # Training arguments that work across versions
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            disable_tqdm=True,
            no_cuda=not torch.cuda.is_available(),
            report_to=None
        )

        # Optimizer that works everywhere
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, None)
        )

        # Start training
        trainer.train()
        trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {'accuracy': accuracy_score(labels, predictions)}

    def classify_subject(self, subject: str) -> str:
        rule_based = self.quick_filter(subject)
        if rule_based:
            return rule_based

        inputs = self.tokenizer(
            subject,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predicted_class = torch.argmax(outputs.logits).item()
        return "recruiter" if predicted_class == 1 else "non-recruiter"

    def batch_classify(self, subjects: list[str]) -> list[str]:
        inputs = self.tokenizer(
            subjects,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_classes = torch.argmax(outputs.logits, dim=-1).tolist()
        return ["recruiter" if pc == 1 else "non-recruiter" for pc in predicted_classes]