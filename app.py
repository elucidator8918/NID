import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import tempfile
import os
import pickle

def convert_pcap_to_csv(pcap_file):
    # Create a temporary file to store the CSV output
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    temp_csv.close()
    
    # Use tshark to convert PCAP to CSV
    cmd = [
        'tshark',
        '-r', pcap_file,
        '-T', 'fields',
        '-E', 'header=y',
        '-E', 'separator=,',
        '-e', 'frame.time_epoch',
        '-e', 'ip.src',
        '-e', 'ip.dst',
        '-e', 'ip.proto',
        '-e', 'tcp.srcport',
        '-e', 'tcp.dstport',
        '-e', 'udp.srcport',
        '-e', 'udp.dstport',
        '-e', 'frame.len'
    ]
    
    with open(temp_csv.name, 'w') as f:
        subprocess.run(cmd, stdout=f)
    
    return temp_csv.name

def process_csv_data(df):
    X = df.drop(['Attack_label', 'Attack_type'], axis=1)
    y = df['Attack_type']
    
    return X, y

def load_model(X, y):    
    # Train model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    return model, X, y, y_pred, y_pred_proba

def plot_confusion_matrix(y_test, y_pred):
    print(y_test, y_pred)
    cm = confusion_matrix(y_test.astype(str), y_pred.astype(str))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def plot_roc_curve(y_test, y_pred_proba):
    # Binarize the output
    n_classes = 15
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    return plt

def main():
    st.title("Network Traffic Analyzer")
    st.write("Upload a PCAP file or a CSV file to analyze network traffic patterns")
    
    uploaded_file = st.file_uploader("Choose a PCAP or CSV file", type=['pcap', 'pcapng', 'csv'])
    
    csv_file = None  # Initialize csv_file to None
    
    if uploaded_file is not None:
        if uploaded_file.type in ['application/vnd.tcpdump.pcap', 'application/octet-stream']:
            # Save uploaded PCAP file temporarily
            temp_pcap = tempfile.NamedTemporaryFile(delete=False, suffix='.pcap')
            temp_pcap.write(uploaded_file.read())
            temp_pcap.close()
            
            with st.spinner("Converting PCAP to CSV..."):
                csv_file = convert_pcap_to_csv(temp_pcap.name)
            
            # Read CSV file from PCAP conversion
            df = pd.read_csv(csv_file)

        else:
            # Read the uploaded CSV file directly
            with st.spinner("Processing CSV data..."):
                df = pd.read_csv(uploaded_file)
        
        # Display the head of the DataFrame
        st.subheader("Data Preview")
        st.write(df.head())  # Display the first few rows of the DataFrame

        # Process the data
        X, y = process_csv_data(df)        
        model, X_test, y_test, y_pred, y_pred_proba = load_model(X, y)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.2%}")
            
            st.subheader("Confusion Matrix")
            cm_plot = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(cm_plot)
        
        with col2:
            st.subheader("ROC Curve")
            roc_plot = plot_roc_curve(y_test, y_pred_proba)
            st.pyplot(roc_plot)
        
        # Cleanup temporary files
        if 'temp_pcap' in locals():
            os.unlink(temp_pcap.name)
        if csv_file:  # Ensure csv_file is defined before unlinking
            os.unlink(csv_file)

if __name__ == "__main__":
    main()