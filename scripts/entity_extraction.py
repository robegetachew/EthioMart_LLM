import time
import torch
from datasets import load_dataset, Dataset
from transformers import XLMRobertaForTokenClassification, DistilBertForTokenClassification, BertForTokenClassification
from transformers import XLMRobertaTokenizer, DistilBertTokenizer, BertTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support

# Load dataset from your local file
def load_ner_dataset(train_file, eval_file):
    # Load data from files
    train_dataset = Dataset.from_text(train_file)
    eval_dataset = Dataset.from_text(eval_file)
    return train_dataset, eval_dataset

# Preprocess data function
def preprocess_function(examples, tokenizer, label2id, max_length=128):
    tokenized_inputs = tokenizer(examples['tokens'], padding='max_length', truncation=True, is_split_into_words=True, max_length=max_length)
    labels = [label2id[label] for label in examples['ner_tags']]  # Map labels to ID
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Metrics function
def compute_metrics(p, label2id):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), axis=-1)
    true_labels = labels

    # Flatten the results
    true_labels_flat = true_labels.flatten()
    predictions_flat = predictions.flatten()

    # Calculate precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flat, predictions_flat, average='macro')
    return {"precision": precision, "recall": recall, "f1": f1}

# Fine-tuning and evaluation function
def fine_tune_and_evaluate(model_name, model, tokenizer, train_dataset, val_dataset, label2id, output_dir):
    train_dataset_enc = train_dataset.map(lambda x: preprocess_function(x, tokenizer, label2id), batched=True)
    val_dataset_enc = val_dataset.map(lambda x: preprocess_function(x, tokenizer, label2id), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f'./logs/{model_name}',
        logging_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_enc,
        eval_dataset=val_dataset_enc,
        compute_metrics=lambda p: compute_metrics(p, label2id),
    )

    # Training the model
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training time for {model_name}: {training_time:.2f} seconds")
    
    return model, trainer

# Define models and tokenizers
def get_models_and_tokenizers(model_name):
    models_and_tokenizers = {
        "xlm-roberta-base": {
            "model": XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=5),  # 5 labels for your dataset
            "tokenizer": XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        },
        "bert-tiny-amharic": {
            "model": BertForTokenClassification.from_pretrained('bert-tiny-amharic', num_labels=5),  # 5 labels for your dataset
            "tokenizer": BertTokenizer.from_pretrained('bert-tiny-amharic')
        },
        "afroxmlr": {
            "model": XLMRobertaForTokenClassification.from_pretrained('afroxmlr', num_labels=5),  # 5 labels for your dataset
            "tokenizer": XLMRobertaTokenizer.from_pretrained('afroxmlr')
        }
    }
    
    return models_and_tokenizers.get(model_name, models_and_tokenizers["xlm-roberta-base"])

# Model comparison function
def compare_models(model_name, train_file, eval_file, output_dir):
    label2id = {'O': 0, 'B-PRODUCT': 1, 'I-PRODUCT': 2, 'I-PRICE': 3, 'I-LOC': 4}  # Modify as per your labels

    # Load datasets
    train_dataset, eval_dataset = load_ner_dataset(train_file, eval_file)

    # Get the appropriate model and tokenizer
    model, tokenizer = get_models_and_tokenizers(model_name)["model"], get_models_and_tokenizers(model_name)["tokenizer"]
    
    print(f"Training and evaluating {model_name}...")
    fine_tuned_model, trainer = fine_tune_and_evaluate(model_name, model, tokenizer, train_dataset, eval_dataset, label2id, output_dir)
    
    # Evaluate the model on the validation set
    eval_result = trainer.evaluate()
    print(f"{model_name} Evaluation result: {eval_result}")
    
    return fine_tuned_model, eval_result
