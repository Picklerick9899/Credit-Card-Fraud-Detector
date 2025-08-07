import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .safe-alert {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, scaler, feature_columns
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def train_demo_model():
    st.info("Training demo model... This may take a moment.")
    
    np.random.seed(42)
    n_samples = 5000
    
    data = {}
    data['Time'] = np.random.uniform(0, 172800, n_samples)
    
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    data['Amount'] = np.random.lognormal(3, 1.5, n_samples)
    
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    data['Class'] = np.zeros(n_samples)
    data['Class'][fraud_indices] = 1
    
    for idx in fraud_indices:
        data['V1'][idx] += np.random.normal(3, 1)
        data['V2'][idx] += np.random.normal(-3, 1)
        data['Amount'][idx] *= np.random.uniform(0.1, 5.0)
    
    df = pd.DataFrame(data)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    with open('fraud_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    st.success("Demo model trained successfully!")
    return model, scaler, X.columns.tolist()

def predict_fraud(model, scaler, feature_columns, data):
    data = data[feature_columns]
    data_scaled = scaler.transform(data)
    predictions = model.predict(data_scaled)
    probabilities = model.predict_proba(data_scaled)
    return predictions, probabilities

def main():
    st.markdown('<h1 class="main-header">🛡️ Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
    st.markdown("**Upload your transaction data to detect potential fraud using advanced machine learning**")
    
    st.sidebar.header("📊 About This App")
    st.sidebar.markdown("""
    This application uses:
    - **Random Forest** classifier
    - **Balanced class weights** for imbalanced data
    - **30 features** for fraud detection
    - **Real-time analysis** of uploaded data
    """)
    
    st.sidebar.header("📋 Expected CSV Format")
    st.sidebar.markdown("""
    Your CSV should contain:
    - `Time`: Transaction time
    - `V1` to `V28`: Anonymized features
    - `Amount`: Transaction amount
    - `Class`: (Optional) Known labels
    """)
    
    model, scaler, feature_columns = load_model()
    
    if model is None:
        st.warning("No trained model found. Training demo model...")
        model, scaler, feature_columns = train_demo_model()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Transaction Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with transaction data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! {len(df)} transactions loaded.")
                
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head())
                
                if 'Class' in df.columns:
                    X = df.drop('Class', axis=1)
                    actual_labels = df['Class']
                else:
                    X = df
                    actual_labels = None
                
                with st.spinner("🔍 Analyzing transactions for fraud..."):
                    predictions, probabilities = predict_fraud(model, scaler, feature_columns, X)
                
                fraud_count = sum(predictions)
                total_transactions = len(predictions)
                fraud_rate = (fraud_count / total_transactions) * 100
                
                st.header("📊 Analysis Results")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Total Transactions", total_transactions)
                
                with col_b:
                    st.metric("Fraudulent Transactions", fraud_count)
                
                with col_c:
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                
                if fraud_count > 0:
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <h3>⚠️ Fraud Detected!</h3>
                        <p>Found <strong>{fraud_count}</strong> potentially fraudulent transactions out of {total_transactions} total transactions.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                        <h3>✅ No Fraud Detected</h3>
                        <p>All {total_transactions} transactions appear to be legitimate.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if fraud_count > 0:
                    st.header("🚨 Fraudulent Transactions")
                    
                    results_df = df.copy()
                    results_df['Predicted_Fraud'] = predictions
                    results_df['Fraud_Probability'] = probabilities[:, 1]
                    
                    fraud_df = results_df[results_df['Predicted_Fraud'] == 1].copy()
                    fraud_df = fraud_df.sort_values('Fraud_Probability', ascending=False)
                    
                    st.dataframe(
                        fraud_df[['Time', 'Amount', 'Fraud_Probability']].style.format({
                            'Fraud_Probability': '{:.4f}',
                            'Amount': '${:.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    csv = fraud_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Fraud Report",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )
                
                if actual_labels is not None:
                    st.header("📈 Model Performance")
                    from sklearn.metrics import accuracy_score, classification_report
                    
                    accuracy = accuracy_score(actual_labels, predictions)
                    st.metric("Model Accuracy", f"{accuracy:.4f}")
                    
                    with st.expander("📊 Detailed Performance Report"):
                        report = classification_report(actual_labels, predictions, output_dict=True)
                        st.json(report)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV file has the correct format.")
    
    with col2:
        st.header("🎯 Quick Test")
        st.markdown("Try with sample data:")
        
        if st.button("🧪 Generate Sample Data"):
            sample_data = {
                'Time': [0.0],
                'Amount': [149.62]
            }
            
            np.random.seed(42)
            for i in range(1, 29):
                sample_data[f'V{i}'] = [np.random.normal(0, 1)]
            
            sample_df = pd.DataFrame(sample_data)
            
            pred, prob = predict_fraud(model, scaler, feature_columns, sample_df)
            
            if pred[0] == 1:
                st.error(f"🚨 FRAUD DETECTED! (Confidence: {prob[0][1]:.4f})")
            else:
                st.success(f"✅ Transaction appears legitimate (Confidence: {prob[0][0]:.4f})")
        
        st.header("ℹ️ How It Works")
        st.markdown("""
        1. **Upload** your CSV file
        2. **AI analyzes** each transaction
        3. **View results** with confidence scores
        4. **Download** detailed fraud report
        """)

if __name__ == "__main__":
    main()
