import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
def load_data(filepath):
    """Load CSV data for training"""
    return pd.read_csv(filepath)

# Preprocess data
def preprocess_data(df, target_column):
    """Split features and target, handle missing values"""
    # Remove rows with missing values
    df = df.dropna()
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical variables if needed
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

# Train model
def train_model(X_train, y_train):
    """Train Random Forest classifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Main workflow
if __name__ == "__main__":
    # 1. Load data
    print("Loading data...")
    df = load_data('data/your_dataset.csv')  # Replace with your dataset path
    
    # 2. Preprocess
    print("Preprocessing data...")
    X, y = preprocess_data(df, target_column='target')  # Replace 'target' with your target column name
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 5. Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # 6. Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\nTraining complete!")
