import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """Load trained model from file"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def make_prediction(model, data):
    """Make predictions on new data"""
    # Ensure data is in correct format
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    
    return prediction, probability

if __name__ == "__main__":
    # Example usage
    model = load_model('model.pkl')
    
    # Make prediction on new data
    new_data = {  # Replace with your features
        'feature1': 5.1,
        'feature2': 3.5,
        'feature3': 1.4,
    }
    
    pred, prob = make_prediction(model, new_data)
    print(f"Prediction: {pred}")
    print(f"Probability: {prob}")
