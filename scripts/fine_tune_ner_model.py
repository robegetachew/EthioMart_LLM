# fine_tune_ner_model.py

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments,
    DataCollatorForTokenClassification
)

class NERFineTuner:
    def __init__(self, model_name, label_list, output_dir="./ner_model"):
        """
        Initialize the NER Fine-Tuner class.

        :param model_name: Pre-trained model name (e.g., 'xlm-roberta-base').
        :param label_list: List of entity labels (e.g., ['O', 'B-ENTITY', 'I-ENTITY']).
        :param output_dir: Directory to save fine-tuned model.
        """
        self.model_name = model_name
        self.label_list = label_list
        self.output_dir = output_dir
        self.label_to_id = {label: i for i, label in enumerate(label_list)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(self.label_list))

    def load_conll_data(self, file_path):
        """
        Load CoNLL-formatted data into a pandas DataFrame.

        :param file_path: Path to the CoNLL file.
        :return: DataFrame with 'tokens' and 'ner_tags'.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        tokens, labels, current_sentence, current_labels = [], [], [], []

        for line in lines:
            if line.strip() == "":
                if current_sentence:
                    tokens.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence, current_labels = [], []
            else:
                token, label = line.strip().split()
                current_sentence.append(token)
                current_labels.append(label)

        return pd.DataFrame({"tokens": tokens, "ner_tags": labels})

    def tokenize_and_align_labels(self, examples):
        """
        Tokenize data and align NER labels with tokens.

        :param examples: Dataset examples containing tokens and ner_tags.
        :return: Tokenized inputs with aligned labels.
        """
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []

        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                else:
                    label_ids.append(self.label_to_id[label[word_idx]] if label[word_idx].startswith("I-") else -100)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_dataset(self, file_path):
        """
        Prepare the dataset by loading and tokenizing the data.

        :param file_path: Path to the CoNLL file.
        :return: Tokenized dataset.
        """
        raw_data = self.load_conll_data(file_path)
        dataset = Dataset.from_pandas(raw_data)
        return dataset.map(self.tokenize_and_align_labels, batched=True)

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics for NER.

        :param pred: Predictions from the model.
        :return: Dictionary of precision, recall, and F1 scores.
        """
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = [[self.id_to_label[label] for label in label_set if label != -100] for label_set in labels]
        true_predictions = [
            [self.id_to_label[pred] for (pred, label) in zip(pred_set, label_set) if label != -100]
            for pred_set, label_set in zip(predictions, labels)
        ]

        report = classification_report(
            [label for sent in true_labels for label in sent],
            [pred for sent in true_predictions for pred in sent],
            output_dict=True,
        )
        return {
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"],
        }

    def fine_tune(self, train_file, eval_file, epochs=3, batch_size=8, learning_rate=5e-5):
        """
        Fine-tune the NER model.

        :param train_file: Path to the training data in CoNLL format.
        :param eval_file: Path to the evaluation data in CoNLL format.
        :param epochs: Number of training epochs.
        :param batch_size: Training batch size.
        :param learning_rate: Learning rate for the optimizer.
        """
        train_dataset = self.prepare_dataset(train_file)
        eval_dataset = self.prepare_dataset(eval_file)

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            logging_steps=500,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_total_limit=2,
            load_best_model_at_end=True,
            logging_dir="./logs",
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model fine-tuned and saved to {self.output_dir}")
