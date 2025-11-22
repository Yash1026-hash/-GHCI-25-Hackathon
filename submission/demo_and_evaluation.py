"""
Training, Evaluation, and Demonstration Script
Orchestrates the full pipeline for the transaction categorizer.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
import json
from transaction_classifier import Categorizer


class TransactionDataset(Dataset):
    """Dataset class for transaction data."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def generate_synthetic_data(taxonomy: dict, num_records=1000) -> pd.DataFrame:
    """
    Create a simulated dataset with realistic transaction strings.
    
    Args:
        taxonomy: Dictionary mapping category IDs to category names
        num_records: Number of synthetic records to generate
        
    Returns:
        DataFrame with raw_string, amount, and target_category_id columns
    """
    categories = taxonomy['categories']
    records = []
    
    # Define realistic transaction patterns for each category
    patterns = {
        1: {  # Fuel
            'strings': [
                'SHELL RETAIL #{}', 'BP SERVICE STATION {}', 'CITGO GAS {}',
                'EXXON MOBIL {}', 'CHEVRON #{}', 'ARCO {}', 'SPEEDWAY {}',
                'VALERO {}', 'MARATHON {}', '7-ELEVEN FUEL {}'
            ],
            'amount_range': (20, 80)
        },
        2: {  # Groceries
            'strings': [
                'WALMART SUPER CENTER {}', 'TARGET STORE {}', 'KROGER #{}',
                'SAFEWAY {}', 'WHOLE FOODS MARKET {}', 'TRADER JOES {}',
                'ALDI {}', 'COSTCO WHOLESALE {}', 'PUBLIX {}', 'WEGMANS {}'
            ],
            'amount_range': (30, 150)
        },
        3: {  # Shopping_General
            'strings': [
                'AMAZON.COM P{}', 'EBAY PAYMENT {}', 'BEST BUY #{}',
                'HOME DEPOT {}', 'MACYS {}', 'NORDSTROM {}', 'KOHLS {}',
                'TARGET.COM {}', 'WALMART.COM {}', 'ETSY {}'
            ],
            'amount_range': (15, 300)
        },
        4: {  # Coffee_Dining
            'strings': [
                'SQ *STARBUCKS #{}', 'MCDONALDS {}', 'SUBWAY {}',
                'CHIPOTLE {}', 'PANERA BREAD {}', 'DUNKIN {}',
                'TIM HORTONS {}', 'COFFEE SHOP {}', 'RESTAURANT {}',
                'SQ *COFFEE {}'
            ],
            'amount_range': (3, 50)
        },
        5: {  # Utilities
            'strings': [
                'ELECTRIC COMPANY {}', 'WATER DEPARTMENT {}', 'GAS COMPANY {}',
                'INTERNET BILL {}', 'PHONE BILL {}', 'CABLE TV {}',
                'UTILITY PAYMENT {}', 'POWER COMPANY {}', 'PG&E {}',
                'CON EDISON {}'
            ],
            'amount_range': (50, 300)
        },
        6: {  # Subscription
            'strings': [
                'NETFLIX {}', 'SPOTIFY {}', 'AMAZON PRIME {}',
                'APPLE MUSIC {}', 'YOUTUBE PREMIUM {}', 'DISNEY+ {}',
                'HULU {}', 'ADOBE CREATIVE CLOUD {}', 'MICROSOFT 365 {}',
                'GYM MEMBERSHIP {}'
            ],
            'amount_range': (5, 30)
        }
    }
    
    # Generate records for each category
    records_per_category = num_records // len(categories)
    
    for category_id, category_name in categories.items():
        if category_id not in patterns:
            continue
            
        pattern_info = patterns[category_id]
        cat_strings = pattern_info['strings']
        amount_min, amount_max = pattern_info['amount_range']
        
        for i in range(records_per_category):
            # Select random pattern
            pattern = np.random.choice(cat_strings)
            
            # Generate transaction ID/number
            tx_id = ''.join([str(np.random.randint(0, 10)) for _ in range(6)])
            
            # Create transaction string
            raw_string = pattern.format(tx_id)
            
            # Add noise variations
            noise_types = [
                lambda s: s,  # No noise
                lambda s: s + ' PURCHASE',
                lambda s: s.replace(' ', ''),
                lambda s: s + ' AUTH',
                lambda s: s.lower(),
                lambda s: 'TXN ' + s,
            ]
            noise_func = np.random.choice(noise_types)
            raw_string = noise_func(raw_string)
            
            # Generate amount
            amount = np.random.uniform(amount_min, amount_max)
            amount = round(amount, 2)
            
            records.append({
                'raw_string': raw_string,
                'amount': amount,
                'target_category_id': category_id - 1,  # 0-indexed for model
                'target_category': category_name
            })
    
    # Shuffle and create DataFrame
    df = pd.DataFrame(records)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df


def _preprocess_for_training(raw_string: str, amount: float) -> str:
    """Preprocess transaction string for training (same as Categorizer._preprocess)."""
    import re
    
    cleaned = raw_string.lower().strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'[^\w\s*#\-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    if amount < 5:
        amount_bin = "AMOUNT_SMALL"
    elif amount < 20:
        amount_bin = "AMOUNT_MEDIUM"
    elif amount < 100:
        amount_bin = "AMOUNT_LARGE"
    else:
        amount_bin = "AMOUNT_VERY_LARGE"
    
    return f"{cleaned} {amount_bin}"


def train_model(df: pd.DataFrame, taxonomy_path: str, model_save_path: str):
    """
    Tokenize data, fine-tune the Transformer model, and save weights.
    
    Args:
        df: Training DataFrame with raw_string, amount, target_category_id
        taxonomy_path: Path to taxonomy YAML
        model_save_path: Path to save the fine-tuned model
    """
    # Load taxonomy to get number of labels
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.safe_load(f)
    num_labels = len(taxonomy['categories'])
    
    # Preprocess text
    print("Preprocessing transaction strings...")
    df['processed_text'] = df.apply(
        lambda row: _preprocess_for_training(row['raw_string'], row['amount']),
        axis=1
    )
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target_category_id'])
    
    # Load tokenizer and model
    print("Loading DistilBERT model...")
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Create datasets
    train_dataset = TransactionDataset(
        train_df['processed_text'].tolist(),
        train_df['target_category_id'].tolist(),
        tokenizer
    )
    val_dataset = TransactionDataset(
        val_df['processed_text'].tolist(),
        val_df['target_category_id'].tolist(),
        tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./training_output',
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        auto_find_batch_size=True,
    )
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predictions, average='macro')
        return {'f1': f1}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {model_save_path}...")
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    torch.save(trainer.model.state_dict(), "best_DistilBERT.pt")
    tokenizer.save_pretrained(model_save_path)
    
    print("Training complete!")


def evaluate_model(model_path: str, test_df: pd.DataFrame, taxonomy_path: str):
    """
    Load trained model and evaluate on test set.
    
    Args:
        model_path: Path to trained model
        test_df: Test DataFrame
        taxonomy_path: Path to taxonomy YAML
    """
    # Load categorizer
    categorizer = Categorizer(model_path, taxonomy_path)
    
    # Make predictions
    print("Running predictions on test set...")
    predictions = []
    true_labels = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        result = categorizer.categorize(row['raw_string'], row['amount'])
        predictions.append(result['category_id'])
        true_labels.append(row['target_category_id'])
    
    # Calculate metrics
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    micro_f1 = f1_score(true_labels, predictions, average='micro')
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nMacro F1-Score: {macro_f1:.4f}")
    print(f"Micro F1-Score: {micro_f1:.4f}")
    
    if macro_f1 > 0.90:
        print("✅ F1-Score meets target (>0.90)")
    else:
        print("⚠️  F1-Score below target (>0.90)")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)
    
    # Classification report
    print("\nPer-Class F1-Score Report:")
    category_names = [categorizer.category_map[i] for i in range(len(categorizer.category_map))]
    
    report_dict = classification_report(
        true_labels,
        predictions,
        target_names=category_names,
        output_dict=True  # Output as dictionary
    )
    print(classification_report(  # Print the text version for console output
        true_labels,
        predictions,
        target_names=category_names,
        output_dict=False
    ))

    # Save test data with predictions
    test_df['predicted_category_id'] = predictions
    test_df['predicted_category'] = test_df['predicted_category_id'].apply(lambda x: categorizer.category_map[x])
    test_df.to_csv('test_data_with_predictions.csv', index=False)
    print("\nTest data with predictions saved to test_data_with_predictions.csv")

    # Save classification report as JSON
    with open('metrics.json', 'w') as f:
        json.dump(report_dict, f, indent=4)
    print("Metrics saved to metrics.json")
    
    
    return macro_f1, predictions, true_labels


def main_demo():
    """Execute the full pipeline: Data generation -> Training -> Evaluation -> Demo."""
    
    taxonomy_path = 'taxonomy.yaml'
    model_save_path = './trained_model'
    
    print("="*60)
    print("AUTONOMOUS AI TRANSACTION CATEGORIZER")
    print("="*60)
    
    # Load taxonomy
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.safe_load(f)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic transaction data...")
    df = generate_synthetic_data(taxonomy, num_records=1500)
    print(f"Generated {len(df)} transaction records")
    print(f"Category distribution:")
    print(df['target_category'].value_counts())
    
    # Split data for training and testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target_category_id'])
    
    # Step 2: Train model
    print("\n[Step 2] Training model...")
    train_model(train_df, taxonomy_path, model_save_path)
    
    # Step 3: Evaluate model
    print("\n[Step 3] Evaluating model...")
    macro_f1, predictions, true_labels = evaluate_model(model_save_path, test_df, taxonomy_path)
    
    # Step 4: Demo predictions
    print("\n" + "="*60)
    print("DEMONSTRATION: Sample Predictions")
    print("="*60)
    
    categorizer = Categorizer(model_save_path, taxonomy_path)
    
    # Sample transactions for demo
    sample_transactions = [
        ("SQ *STARBUCKS #1234", 5.50),
        ("AMAZON.COM P1X5G8", 89.99),
        ("SHELL RETAIL 4567", 45.20),
        ("WALMART SUPER CENTER 789", 125.30),
        ("NETFLIX", 15.99),
    ]
    
    print("\nPredictions on sample transactions:")
    for raw_string, amount in sample_transactions:
        result = categorizer.categorize(raw_string, amount)
        print(f"\nTransaction: {raw_string} | Amount: ${amount}")
        print(f"  → Category: {result['category']}")
        print(f"  → Confidence: {result['confidence']:.4f}")
    
    # Step 5: Explainability demo
    print("\n" + "="*60)
    print("EXPLAINABILITY: SHAP Analysis")
    print("="*60)
    
    # Use a complex transaction for explanation
    complex_transaction = ("SQ *STARBUCKS #1234 DOWNTOWN LOCATION", 12.50)
    explanation = categorizer.explain_prediction(
        complex_transaction[0],
        complex_transaction[1],
        top_k=10
    )
    
    print(f"\nTransaction: {complex_transaction[0]} | Amount: ${complex_transaction[1]}")
    print(f"Predicted Category: {explanation['predicted_category']}")
    print(f"Confidence: {explanation['confidence']:.4f}")
    print("\nTop Contributing Words:")
    for i, attr in enumerate(explanation['word_attributions'], 1):
        sign = "+" if attr['score'] > 0 else ""
        print(f"  {i}. {attr['word']:20s} {sign}{attr['score']:8.4f}")
    
    # Step 6: Feedback logging demo
    print("\n" + "="*60)
    print("FEEDBACK LOGGING DEMO")
    print("="*60)
    print("\nLogging feedback corrections...")
    categorizer.log_feedback("TX001", "AMAZON.COM", "Shopping_General")
    categorizer.log_feedback("TX002", "STARBUCKS", "Coffee_Dining")
    print("Feedback logged to feedback.log")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main_demo()

