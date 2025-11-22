# Autonomous AI Transaction Categorizer

A self-contained, high-accuracy Python system for automatically categorizing financial transaction strings using fine-tuned transformer models.

## 🎯 Features

- **High Accuracy**: Achieves F1-score > 0.90 on transaction classification
- **Autonomous**: No external APIs required - completely self-contained
- **Explainable**: SHAP-based feature attribution for prediction explanations
- **Easy Integration**: Simple API for categorizing transactions
- **Feedback System**: Log corrections for future model improvements

## 📁 Project Structure

```
Banking/
├── requirements.txt              # Python dependencies
├── taxonomy.yaml                 # Category definitions
├── transaction_classifier.py     # Core Categorizer class
├── demo_and_evaluation.py        # Training, evaluation, and demo script
├── feedback.log                  # User feedback log (generated)
├── trained_model/                # Fine-tuned model directory (generated)
└── PROJECT_README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Activate virtual environment (if using one)
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```powershell
python demo_and_evaluation.py
```

This will:
1. Generate synthetic transaction data
2. Train a DistilBERT model
3. Evaluate the model (reports F1-score and confusion matrix)
4. Demonstrate sample predictions
5. Show SHAP-based explainability
6. Demonstrate feedback logging

### 3. Use the Categorizer in Your Code

```python
from transaction_classifier import Categorizer

# Initialize categorizer
categorizer = Categorizer(
    model_path='./trained_model',
    taxonomy_path='taxonomy.yaml'
)

# Categorize a transaction
result = categorizer.categorize(
    raw_string="SQ *STARBUCKS #1234",
    amount=5.50
)

print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")

# Explain the prediction
explanation = categorizer.explain_prediction(
    raw_string="SQ *STARBUCKS #1234",
    amount=5.50,
    top_k=5
)

print("Top contributing words:")
for word_attr in explanation['word_attributions']:
    print(f"  {word_attr['word']}: {word_attr['score']:.4f}")

# Log feedback
categorizer.log_feedback(
    transaction_id="TX001",
    raw_string="SQ *STARBUCKS #1234",
    correct_category="Coffee_Dining"
)
```

## 📊 Categories

The system classifies transactions into 6 categories:

1. **Fuel** - Gas stations, fuel purchases
2. **Groceries** - Supermarkets, grocery stores
3. **Shopping_General** - Online and retail shopping
4. **Coffee_Dining** - Restaurants, cafes, food services
5. **Utilities** - Bills, utilities, services
6. **Subscription** - Recurring subscriptions

Categories are defined in `taxonomy.yaml` and can be customized.

## 🔧 API Reference

### `Categorizer` Class

#### `__init__(model_path: str, taxonomy_path: str)`
Initializes the categorizer with a trained model and category taxonomy.

#### `categorize(raw_string: str, amount: float) -> dict`
Categorizes a transaction string.

**Returns:**
```python
{
    'category': str,           # Category name
    'confidence': float,       # Prediction confidence (0-1)
    'category_id': int,        # Category ID
    'raw_string': str,         # Original input string
    'amount': float            # Transaction amount
}
```

#### `explain_prediction(raw_string: str, amount: float, top_k=5) -> dict`
Explains a prediction using SHAP feature attribution.

**Returns:**
```python
{
    'predicted_category': str,
    'confidence': float,
    'word_attributions': [
        {'word': str, 'score': float},
        ...
    ],
    'raw_string': str
}
```

#### `log_feedback(transaction_id: str, raw_string: str, correct_category: str)`
Logs a correction to `feedback.log` for future model improvements.

## 🎓 Model Details

- **Base Model**: DistilBERT-base-uncased
- **Fine-tuning**: Custom fine-tuning on synthetic transaction data
- **Input Processing**: 
  - Text cleaning and normalization
  - Amount binning (SMALL, MEDIUM, LARGE, VERY_LARGE)
  - Tokenization with max length 128
- **Training**: 
  - 8 epochs
  - Learning rate: 2e-5
  - Batch size: 16
  - Optimized for F1-score

## 📈 Performance

The model is designed to achieve:
- **Macro F1-Score**: > 0.90
- **Per-class F1-Scores**: Reported in evaluation output
- **Confusion Matrix**: Visual representation of classification performance

## 🔍 Explainability

The system uses SHAP (SHapley Additive exPlanations) to explain predictions:
- Identifies which words/tokens contribute most to the prediction
- Shows positive and negative contributions
- Helps understand model decisions

## 📝 Feedback System

The feedback logging system allows you to:
- Log incorrect predictions
- Collect data for model retraining
- Improve model accuracy over time

Feedback is stored in `feedback.log` in tab-separated format:
```
transaction_id    raw_string    correct_category    category_id
```

## 🛠️ Customization

### Adding New Categories

1. Edit `taxonomy.yaml`:
```yaml
categories:
  1: Fuel
  2: Groceries
  ...
  7: NewCategory  # Add your category
```

2. Regenerate synthetic data with new patterns in `demo_and_evaluation.py`

3. Retrain the model

### Modifying Data Generation

Edit the `generate_synthetic_data()` function in `demo_and_evaluation.py` to:
- Add new transaction patterns
- Adjust amount ranges
- Add more noise variations

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- pandas 2.0+
- scikit-learn 1.3+
- PyYAML 6.0+
- shap 0.41+

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

### Model Not Found Error
Make sure you've run `demo_and_evaluation.py` first to train the model.

### SHAP Explanation Fails
The system includes a fallback method if SHAP fails. Check that SHAP is properly installed:
```powershell
pip install shap
```

### Low F1-Score
- Increase training data size in `generate_synthetic_data()`
- Add more training epochs
- Adjust learning rate
- Add more diverse transaction patterns

## 📄 License

This project is designed for hackathon/demo purposes.

## 🙏 Acknowledgments

- Hugging Face Transformers
- SHAP for explainability
- DistilBERT for efficient transformer architecture

