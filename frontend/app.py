import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn_model.model import CNNIDS

st.set_page_config(page_title="Federated Learning IDS", layout="wide")

st.title("🛡️ FL-Based Intrusion Detection System")
st.markdown("### Using Chimp Optimization & CNN with Federated Learning")

# Sidebar
page = st.sidebar.selectbox("Navigation", ["Prediction", "Model Analysis", "Training History"])

@st.cache_resource
def load_resources():
    try:
        # Load metadata
        classes = np.load("models/classes.npy", allow_pickle=True)
        selected_mask = np.load("models/selected_features.npy")
        scaler = joblib.load("models/scaler.pkl")
        
        # Determine feature count
        n_features_selected = np.sum(selected_mask)
        num_classes = len(classes)
        
        # Load Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CNNIDS(input_dim=n_features_selected, num_classes=num_classes)
        model.load_state_dict(torch.load("models/best_global_model.pth", map_location=device))
        model.to(device)
        model.eval()
        
        return model, scaler, selected_mask, classes, device
    except FileNotFoundError as e:
        return None, None, None, None, None

model, scaler, selected_mask, classes, device = load_resources()

if model is None:
    st.error("Model artifacts not found! Please run 'main_train.py' first.")
    st.stop()

if page == "Prediction":
    st.header("Upload Traffic Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:", df.head())
            
            # Preprocessing
            # 1. Cleaning similar to training
            df.columns = df.columns.str.strip()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            # 2. Handle numeric
            # Drop Label if present to simulate inference data
            if 'Label' in df.columns:
                X_raw = df.drop(columns=['Label'])
                y_true = df['Label'] # Only if we want to compare
            else:
                X_raw = df
            
            # Select numeric only
            X_numeric = X_raw.select_dtypes(include=[np.number])
            
            # Ensure columns match scaler (this is tricky with raw CSVs)
            # We assume user uploads compatible CSV (subset of CICIDS)
            # If shape mismatch, we try to fit valid columns or error out.
            
            if X_numeric.shape[1] != scaler.n_features_in_:
                st.error(f"Feature mismatch! Expected {scaler.n_features_in_}, got {X_numeric.shape[1]}. Please ensure columns match training data.")
            else:
                # Scale
                X_scaled = scaler.transform(X_numeric)
                
                # Feature Selection
                X_selected = X_scaled[:, selected_mask == 1]
                
                # Prediction
                st.info("Running Inference...")
                tensor_X = torch.tensor(X_selected, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    outputs = model(tensor_X)
                    _, preds = torch.max(outputs, 1)
                
                # Decode predictions
                pred_labels = [classes[p] for p in preds.cpu().numpy()]
                
                # Display Results
                df['Prediction'] = pred_labels
                st.success("Analysis Complete!")
                st.dataframe(df)
                
                # Distribution Plot
                st.subheader("Attack Class Distribution")
                counts = pd.Series(pred_labels).value_counts()
                st.bar_chart(counts)
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "Model Analysis":
    st.header("Feature Selection (ChOA)")
    st.write(f"Total Features: {len(selected_mask)}")
    st.write(f"Selected Features: {np.sum(selected_mask)}")
    
    # Visualize mask
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.heatmap([selected_mask], cmap="Greens", cbar=False, yticklabels=False)
    ax.set_title("Selected Features Mask (Green = Selected)")
    st.pyplot(fig)
    
    st.header("Confusion Matrix (Test Set)")
    if os.path.exists("evaluation/confusion_matrix.png"):
        st.image("evaluation/confusion_matrix.png")
    else:
        st.warning("Confusion matrix not found.")

elif page == "Training History":
    st.header("Federated Learning Performance")
    if os.path.exists("evaluation/training_history.csv"):
        history = pd.read_csv("evaluation/training_history.csv")
        st.dataframe(history)
        
        st.subheader("Accuracy over Rounds")
        st.line_chart(history.set_index('round')['accuracy'])
        
        st.subheader("Metrics Trend")
        st.line_chart(history.set_index('round')[['precision', 'recall', 'f1']])
    else:
        st.warning("Training history not found.")
