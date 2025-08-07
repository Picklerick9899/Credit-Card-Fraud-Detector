import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_demo_model():
    print("Creating demo fraud detection model...")
    
    np.random.seed(42)
    n_samples = 5000
    
    data = {}
    data['Time'] = np.random.uniform(0, 172800, n_samples)
    
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    data['Amount'] = np.random.lognormal(3, 1.5, n_samples)
    
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.002), replace=False)
    data['Class'] = np.zeros(n_samples)
    data['Class'][fraud_indices] = 1
    
    for idx in fraud_indices:
        data['V1'][idx] += np.random.normal(2, 0.5)
        data['V2'][idx] += np.random.normal(-2, 0.5)
        data['Amount'][idx] *= np.random.uniform(0.1, 3.0)
    
    df = pd.DataFrame(data)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"Dataset created: {len(df)} transactions, {sum(y)} fraudulent")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    os.makedirs('model', exist_ok=True)
    
    with open('fraud_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    print("\nModel saved successfully!")
    print("Files created:")
    print("- fraud_model.pkl")
    print("- scaler.pkl") 
    print("- feature_columns.pkl")
    
    return model, scaler, X.columns.tolist()

if __name__ == "__main__":
    create_demo_model()
    print("\nDemo model training completed!")
    print("You can now run the Streamlit app!")
