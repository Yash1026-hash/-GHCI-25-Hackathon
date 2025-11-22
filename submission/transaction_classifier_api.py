"""
API-Based Transaction Categorizer
Uses external APIs to automatically categorize transactions based on internet data.
"""

import re
import yaml
import requests
import json
from typing import Dict, Optional
from urllib.parse import quote


class APICategorizer:
    """API-based transaction categorizer using internet data."""
    
    def __init__(self, taxonomy_path: str, api_key: Optional[str] = None, api_provider: str = "plaid"):
        """
        Initializes the API categorizer.
        
        Args:
            taxonomy_path: Path to the taxonomy YAML file
            api_key: API key for the categorization service (optional for some providers)
            api_provider: API provider to use ('plaid', 'openai', 'web_lookup', 'hybrid')
        """
        # Load taxonomy map from YAML
        with open(taxonomy_path, 'r') as f:
            taxonomy = yaml.safe_load(f)
        self.category_map = {int(k): v for k, v in taxonomy['categories'].items()}
        self.reverse_category_map = {v: k for k, v in self.category_map.items()}
        
        self.api_key = api_key
        self.api_provider = api_provider.lower()
        
        # Mapping from API categories to our taxonomy
        self.category_mapping = self._build_category_mapping()
        
        # Merchant database (can be expanded with web lookups)
        self.merchant_db = self._load_merchant_database()
    
    def _build_category_mapping(self) -> Dict[str, str]:
        """Build mapping from external API categories to our taxonomy."""
        return {
            # Plaid categories
            'gas_stations': 'Fuel',
            'gas': 'Fuel',
            'fuel': 'Fuel',
            'groceries': 'Groceries',
            'supermarkets': 'Groceries',
            'food_and_drink': 'Groceries',
            'general_merchandise': 'Shopping_General',
            'shops': 'Shopping_General',
            'general': 'Shopping_General',
            'restaurants': 'Coffee_Dining',
            'fast_food': 'Coffee_Dining',
            'food_drink': 'Coffee_Dining',
            'coffee_shops': 'Coffee_Dining',
            'utilities': 'Utilities',
            'home': 'Utilities',
            'services': 'Utilities',
            'recurring': 'Subscription',
            'subscriptions': 'Subscription',
            'entertainment': 'Subscription',
            'software': 'Subscription',
            
            # Generic mappings
            'fuel': 'Fuel',
            'grocery': 'Groceries',
            'shopping': 'Shopping_General',
            'dining': 'Coffee_Dining',
            'utility': 'Utilities',
            'subscription': 'Subscription',
        }
    
    def _load_merchant_database(self) -> Dict[str, str]:
        """Load merchant database with known merchants and their categories."""
        return {
            # Fuel
            'shell': 'Fuel', 'bp': 'Fuel', 'citgo': 'Fuel', 'exxon': 'Fuel',
            'mobil': 'Fuel', 'chevron': 'Fuel', 'arco': 'Fuel', 'speedway': 'Fuel',
            'valero': 'Fuel', 'marathon': 'Fuel', '7-eleven fuel': 'Fuel',
            
            # Groceries
            'walmart': 'Groceries', 'target': 'Groceries', 'kroger': 'Groceries',
            'safeway': 'Groceries', 'whole foods': 'Groceries', 'trader joes': 'Groceries',
            'aldi': 'Groceries', 'costco': 'Groceries', 'publix': 'Groceries',
            'wegmans': 'Groceries',
            
            # Shopping
            'amazon': 'Shopping_General', 'ebay': 'Shopping_General',
            'best buy': 'Shopping_General', 'home depot': 'Shopping_General',
            'macys': 'Shopping_General', 'nordstrom': 'Shopping_General',
            'kohls': 'Shopping_General', 'etsy': 'Shopping_General',
            
            # Coffee/Dining
            'starbucks': 'Coffee_Dining', 'mcdonalds': 'Coffee_Dining',
            'subway': 'Coffee_Dining', 'chipotle': 'Coffee_Dining',
            'panera': 'Coffee_Dining', 'dunkin': 'Coffee_Dining',
            'tim hortons': 'Coffee_Dining',
            
            # Utilities
            'electric company': 'Utilities', 'water department': 'Utilities',
            'gas company': 'Utilities', 'internet bill': 'Utilities',
            'phone bill': 'Utilities', 'cable tv': 'Utilities',
            'power company': 'Utilities', 'pg&e': 'Utilities',
            'con edison': 'Utilities',
            
            # Subscriptions
            'netflix': 'Subscription', 'spotify': 'Subscription',
            'amazon prime': 'Subscription', 'apple music': 'Subscription',
            'youtube premium': 'Subscription', 'disney+': 'Subscription',
            'hulu': 'Subscription', 'adobe': 'Subscription',
            'microsoft 365': 'Subscription', 'gym membership': 'Subscription',
        }
    
    def _preprocess_merchant_name(self, raw_string: str) -> str:
        """Clean and normalize merchant name for lookup."""
        # Remove common prefixes/suffixes
        cleaned = raw_string.upper()
        cleaned = re.sub(r'^(SQ\s*\*|TXN\s*|AUTH\s*)', '', cleaned)
        cleaned = re.sub(r'#[0-9]+', '', cleaned)
        cleaned = re.sub(r'P[0-9A-Z]+', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = cleaned.lower()
        
        # Extract merchant name (before first # or special char)
        merchant_match = re.match(r'^([^#*0-9]+)', cleaned)
        if merchant_match:
            cleaned = merchant_match.group(1).strip()
        
        return cleaned
    
    def _lookup_merchant_local(self, merchant_name: str) -> Optional[str]:
        """Look up merchant in local database."""
        merchant_lower = merchant_name.lower()
        
        # Exact match
        if merchant_lower in self.merchant_db:
            return self.merchant_db[merchant_lower]
        
        # Partial match
        for key, category in self.merchant_db.items():
            if key in merchant_lower or merchant_lower in key:
                return category
        
        return None
    
    def _categorize_with_plaid_api(self, raw_string: str, amount: float) -> Optional[Dict]:
        """Categorize using Plaid Enrich API (requires API key)."""
        if not self.api_key:
            return None
        
        try:
            url = "https://production.plaid.com/enrich/transactions/enrich"
            headers = {
                "Content-Type": "application/json",
                "PLAID-CLIENT-ID": self.api_key.split(":")[0] if ":" in self.api_key else "",
                "PLAID-SECRET": self.api_key.split(":")[1] if ":" in self.api_key else self.api_key
            }
            
            payload = {
                "client_transaction_id": "txn_001",
                "description": raw_string,
                "amount": amount,
                "account_type": "depository"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                category = data.get('category', [])
                if category:
                    # Map Plaid category to our taxonomy
                    plaid_category = category[0].get('primary', '').lower()
                    mapped_category = self.category_mapping.get(plaid_category)
                    if mapped_category:
                        return {
                            'category': mapped_category,
                            'confidence': 0.95,
                            'merchant_name': data.get('merchant_name', raw_string),
                            'source': 'plaid_api'
                        }
        except Exception as e:
            print(f"Plaid API error: {e}")
        
        return None
    
    def _categorize_with_openai_api(self, raw_string: str, amount: float) -> Optional[Dict]:
        """Categorize using OpenAI API for intelligent classification."""
        if not self.api_key:
            return None
        
        try:
            import openai
            
            prompt = f"""Categorize this financial transaction into one of these categories:
Fuel, Groceries, Shopping_General, Coffee_Dining, Utilities, Subscription

Transaction: {raw_string}
Amount: ${amount}

Return only the category name, nothing else."""
            
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial transaction categorizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0
            )
            
            category = response.choices[0].message.content.strip()
            
            # Map to our taxonomy
            if category in self.reverse_category_map:
                return {
                    'category': category,
                    'confidence': 0.90,
                    'source': 'openai_api'
                }
        except Exception as e:
            print(f"OpenAI API error: {e}")
        
        return None
    
    def _categorize_with_web_lookup(self, raw_string: str, amount: float) -> Optional[Dict]:
        """Categorize using web lookup and pattern matching."""
        # Clean merchant name
        merchant_name = self._preprocess_merchant_name(raw_string)
        
        # Try local database first
        category = self._lookup_merchant_local(merchant_name)
        if category:
            return {
                'category': category,
                'confidence': 0.85,
                'merchant_name': merchant_name,
                'source': 'local_db'
            }
        
        # Pattern-based classification
        merchant_lower = merchant_name.lower()
        
        # Fuel patterns
        if any(word in merchant_lower for word in ['shell', 'bp', 'gas', 'fuel', 'exxon', 'chevron', 'arco', 'speedway', 'valero', 'marathon']):
            return {'category': 'Fuel', 'confidence': 0.80, 'source': 'pattern_match'}
        
        # Groceries patterns
        if any(word in merchant_lower for word in ['walmart', 'target', 'kroger', 'safeway', 'whole foods', 'grocery', 'supermarket', 'costco']):
            return {'category': 'Groceries', 'confidence': 0.80, 'source': 'pattern_match'}
        
        # Shopping patterns
        if any(word in merchant_lower for word in ['amazon', 'ebay', 'best buy', 'home depot', 'macys', 'nordstrom', 'shopping', 'retail']):
            return {'category': 'Shopping_General', 'confidence': 0.80, 'source': 'pattern_match'}
        
        # Coffee/Dining patterns
        if any(word in merchant_lower for word in ['starbucks', 'mcdonalds', 'subway', 'chipotle', 'panera', 'dunkin', 'restaurant', 'cafe', 'coffee', 'dining']):
            return {'category': 'Coffee_Dining', 'confidence': 0.80, 'source': 'pattern_match'}
        
        # Utilities patterns
        if any(word in merchant_lower for word in ['electric', 'water', 'gas company', 'utility', 'power', 'bill', 'internet', 'phone', 'cable']):
            return {'category': 'Utilities', 'confidence': 0.80, 'source': 'pattern_match'}
        
        # Subscription patterns
        if any(word in merchant_lower for word in ['netflix', 'spotify', 'prime', 'disney', 'hulu', 'adobe', 'microsoft', 'subscription', 'membership']):
            return {'category': 'Subscription', 'confidence': 0.80, 'source': 'pattern_match'}
        
        # Amount-based heuristics
        if amount < 5:
            return {'category': 'Coffee_Dining', 'confidence': 0.60, 'source': 'amount_heuristic'}
        elif amount < 20:
            return {'category': 'Coffee_Dining', 'confidence': 0.55, 'source': 'amount_heuristic'}
        elif amount < 50:
            return {'category': 'Groceries', 'confidence': 0.55, 'source': 'amount_heuristic'}
        elif amount < 100:
            return {'category': 'Shopping_General', 'confidence': 0.55, 'source': 'amount_heuristic'}
        else:
            return {'category': 'Utilities', 'confidence': 0.55, 'source': 'amount_heuristic'}
    
    def categorize(self, raw_string: str, amount: float) -> Dict:
        """
        Categorize transaction using API or web lookup.
        
        Args:
            raw_string: Raw transaction description
            amount: Transaction amount
            
        Returns:
            Dictionary with category, confidence, and metadata
        """
        result = None
        
        # Try API-based categorization based on provider
        if self.api_provider == 'plaid' and self.api_key:
            result = self._categorize_with_plaid_api(raw_string, amount)
        elif self.api_provider == 'openai' and self.api_key:
            result = self._categorize_with_openai_api(raw_string, amount)
        elif self.api_provider == 'hybrid':
            # Try Plaid first, then OpenAI, then web lookup
            if self.api_key:
                result = self._categorize_with_plaid_api(raw_string, amount)
                if not result:
                    result = self._categorize_with_openai_api(raw_string, amount)
            if not result:
                result = self._categorize_with_web_lookup(raw_string, amount)
        else:
            # Default to web lookup
            result = self._categorize_with_web_lookup(raw_string, amount)
        
        # Fallback to web lookup if API fails
        if not result:
            result = self._categorize_with_web_lookup(raw_string, amount)
        
        # Get category ID
        category_id = self.reverse_category_map.get(result['category'], -1)
        
        return {
            'category': result['category'],
            'confidence': result['confidence'],
            'category_id': category_id,
            'raw_string': raw_string,
            'amount': amount,
            'source': result.get('source', 'unknown'),
            'merchant_name': result.get('merchant_name', raw_string)
        }
    
    def explain_prediction(self, raw_string: str, amount: float, top_k=5) -> Dict:
        """
        Explain the prediction (simplified for API-based approach).
        
        Args:
            raw_string: Raw transaction description
            amount: Transaction amount
            top_k: Number of top features to return
            
        Returns:
            Dictionary containing prediction and explanation
        """
        prediction = self.categorize(raw_string, amount)
        
        # Extract key words that led to categorization
        merchant_name = self._preprocess_merchant_name(raw_string)
        words = merchant_name.split()
        
        # Identify important words based on merchant database
        important_words = []
        for word in words:
            if len(word) > 2:  # Skip short words
                # Check if word is in merchant database
                for key, category in self.merchant_db.items():
                    if word in key and category == prediction['category']:
                        important_words.append({
                            'word': word,
                            'score': 1.0,
                            'reason': 'Merchant match'
                        })
                        break
        
        # If no important words found, use all words with equal weight
        if not important_words:
            for word in words[:top_k]:
                if len(word) > 2:
                    important_words.append({
                        'word': word,
                        'score': 0.5,
                        'reason': 'Pattern match'
                    })
        
        return {
            'predicted_category': prediction['category'],
            'confidence': prediction['confidence'],
            'word_attributions': important_words[:top_k],
            'raw_string': raw_string,
            'source': prediction.get('source', 'unknown')
        }
    
    def log_feedback(self, transaction_id: str, raw_string: str, correct_category: str):
        """
        Log feedback for future improvements.
        
        Args:
            transaction_id: Unique transaction identifier
            raw_string: Raw transaction description
            correct_category: Correct category label
        """
        category_id = self.reverse_category_map.get(correct_category, -1)
        log_entry = f"{transaction_id}\t{raw_string}\t{correct_category}\t{category_id}\n"
        
        with open('feedback.log', 'a', encoding='utf-8') as f:
            f.write(log_entry)




