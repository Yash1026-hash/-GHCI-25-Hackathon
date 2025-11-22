"""
Autonomous AI Transaction Categorizer
Core library for classifying financial transaction strings.
"""

import re
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import shap


class Categorizer:
    """Self-contained transaction categorizer with explainability."""
    
    def __init__(self, model_path: str, taxonomy_path: str):
        """
        Initializes the model, tokenizer, and category map.
        
        Args:
            model_path: Path to the fine-tuned model directory
            taxonomy_path: Path to the taxonomy YAML file
        """
        # Load taxonomy map from YAML
        with open(taxonomy_path, 'r') as f:
            taxonomy = yaml.safe_load(f)
        self.category_map = {int(k) - 1: v for k, v in taxonomy['categories'].items()}
        self.reverse_category_map = {v: k for k, v in self.category_map.items()}
        self.num_labels = len(self.category_map)
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        
        # Load fine-tuned model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_labels
        )
        self.model.eval()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _preprocess(self, raw_string: str, amount: float) -> str:
        """
        Cleans the string and incorporates the amount feature.
        
        Args:
            raw_string: Raw transaction description
            amount: Transaction amount
            
        Returns:
            Preprocessed input string
        """
        # Clean noise: lowercase, remove extra whitespace
        cleaned = raw_string.lower().strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove or normalize special characters (keep basic punctuation)
        cleaned = re.sub(r'[^\w\s*#\-]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Convert amount to bin feature
        # Create meaningful bins for transaction amounts
        if amount < 5:
            amount_bin = "AMOUNT_SMALL"
        elif amount < 20:
            amount_bin = "AMOUNT_MEDIUM"
        elif amount < 100:
            amount_bin = "AMOUNT_LARGE"
        else:
            amount_bin = "AMOUNT_VERY_LARGE"
        
        # Concatenate amount feature with cleaned string
        processed = f"{cleaned} {amount_bin}"
        
        return processed
    
    def categorize(self, raw_string: str, amount: float) -> dict:
        """
        Performs inference and returns the result.
        
        Args:
            raw_string: Raw transaction description
            amount: Transaction amount
            
        Returns:
            Dictionary with category, confidence, raw_string, and amount
        """
        # Preprocess input
        processed_text = self._preprocess(raw_string, amount)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get predicted category
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_id].item()
        category = self.category_map[predicted_id]
        
        return {
            'category': category,
            'confidence': float(confidence),
            'category_id': predicted_id,
            'raw_string': raw_string,
            'amount': amount
        }
    
    def explain_prediction(self, raw_string: str, amount: float, top_k=5) -> dict:
        """
        Uses SHAP to determine feature importance for the prediction.
        
        Args:
            raw_string: Raw transaction description
            amount: Transaction amount
            top_k: Number of top words to return
            
        Returns:
            Dictionary containing predicted category and word attributions
        """
        # Get prediction first
        prediction = self.categorize(raw_string, amount)
        predicted_id = prediction['category_id']
        
        # Preprocess input
        processed_text = self._preprocess(raw_string, amount)
        
        # Create SHAP explainer wrapper
        def model_wrapper(texts):
            """Wrapper function for SHAP that returns logits for the predicted class."""
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            if isinstance(texts, str):
                texts = [texts]
            
            batch_inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                logits = outputs.logits
                # Return logits for the predicted class only
                return logits[:, predicted_id].cpu().tolist()
        
        try:
            # Use SHAP's Explainer with text masker
            masker = shap.maskers.Text(self.tokenizer, mask_token='[MASK]')
            explainer = shap.Explainer(model_wrapper, masker)
            
            # Get SHAP values
            shap_values = explainer([processed_text], silent=True)
            
            # Extract values and tokens
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) > 1:
                    scores = shap_values.values[0]
                else:
                    scores = shap_values.values
            else:
                scores = shap_values[0].values if hasattr(shap_values[0], 'values') else shap_values[0]
            
            # Get tokenized text
            tokens = self.tokenizer.tokenize(processed_text)
            if len(tokens) > len(scores):
                tokens = tokens[:len(scores)]
            elif len(scores) > len(tokens):
                scores = scores[:len(tokens)]
            
            # Create word-score pairs
            word_attributions = []
            for token, score in zip(tokens, scores):
                if token not in ['[CLS]', '[SEP]', '[PAD]', '[MASK]']:
                    word_attributions.append({
                        'word': token,
                        'score': float(score)
                    })
            
            # Sort by absolute score and get top_k
            word_attributions.sort(key=lambda x: abs(x['score']), reverse=True)
            top_attributions = word_attributions[:top_k] if len(word_attributions) >= top_k else word_attributions
            
        except Exception as e:
            # Fallback: Simple token-based attribution if SHAP fails
            print(f"Warning: SHAP explanation failed ({e}), using fallback method")
            tokens = processed_text.split()
            # Simple heuristic: give equal weight to tokens
            word_attributions = [
                {'word': token, 'score': 1.0 / len(tokens)} 
                for token in tokens[:top_k]
            ]
            top_attributions = word_attributions
        
        return {
            'predicted_category': prediction['category'],
            'confidence': prediction['confidence'],
            'word_attributions': top_attributions,
            'raw_string': raw_string
        }
    
    def log_feedback(self, transaction_id: str, raw_string: str, correct_category: str):
        """
        Simulates logging correction data to a file for future fine-tuning.
        
        Args:
            transaction_id: Unique transaction identifier
            raw_string: Raw transaction description
            correct_category: Correct category label
        """
        # Get category ID if it exists
        category_id = self.reverse_category_map.get(correct_category, -1)
        
        # Format log entry
        log_entry = f"{transaction_id}\t{raw_string}\t{correct_category}\t{category_id}\n"
        
        # Append to feedback log
        with open('feedback.log', 'a', encoding='utf-8') as f:
            f.write(log_entry)

