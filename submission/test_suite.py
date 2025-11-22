
import pytest
import os
import subprocess
from transaction_classifier_api import APICategorizer
from transaction_classifier import Categorizer

MODEL_PATH = './trained_model'
TAXONOMY_PATH = 'taxonomy.yaml'

@pytest.fixture(scope="module")
def trained_model():
    """Fixture to ensure the model is trained before tests run."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Training model...")
        # Use py.exe from venv to run the script
        subprocess.run([".\\venv\\Scripts\\python.exe", "demo_and_evaluation.py"], check=True)
    return MODEL_PATH

@pytest.fixture
def api_categorizer():
    """Fixture for the APICategorizer."""
    return APICategorizer(taxonomy_path=TAXONOMY_PATH, api_provider='web_lookup')

@pytest.fixture
def local_categorizer(trained_model):
    """Fixture for the local Categorizer."""
    return Categorizer(model_path=trained_model, taxonomy_path=TAXONOMY_PATH)

def test_api_categorizer_web_lookup(api_categorizer):
    """Test the APICategorizer with the web_lookup provider."""
    result = api_categorizer.categorize("STARBUCKS", 10.0)
    assert result['category'] == "Coffee_Dining"
    assert result['confidence'] > 0.8
    assert result['source'] == 'local_db'

def test_local_categorizer_initialization(local_categorizer):
    """Test if the local categorizer can be initialized."""
    assert local_categorizer is not None

def test_local_categorizer_prediction(local_categorizer):
    """Test a prediction with the local categorizer."""
    result = local_categorizer.categorize("WALMART", 100.0)
    assert result['category'] == "Groceries"
    assert result['confidence'] > 0.5 # Confidence may vary

def test_explainability(local_categorizer):
    """Test the explainability feature of the local categorizer."""
    explanation = local_categorizer.explain_prediction("SQ *STARBUCKS #1234 DOWNTOWN", 12.50)
    assert explanation['predicted_category'] == 'Coffee_Dining'
    assert 'word_attributions' in explanation
    assert len(explanation['word_attributions']) > 0

# More tests can be added for edge cases and other API providers.
