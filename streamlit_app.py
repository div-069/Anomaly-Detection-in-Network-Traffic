# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="Network Anomaly Detection", layout="centered")
st.markdown("<h1 style='text-align:center;'>Network Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)

st.sidebar.title("⚙️ Choose Model")
model_choice = st.sidebar.radio(
    "Select which model to use:",
    ["Autoencoder (Deep Learning)", "Isolation Forest (Classical ML)"],
    index=0
)

if model_choice == "Autoencoder (Deep Learning)":
    st.sidebar.info("✅ Better for complex patterns; unsupervised deep learning.")
else:
    st.sidebar.info("⚡ Faster; good for general anomaly detection.")

uploaded = st.file_uploader("📂 Upload test CSV file (with or without labels):")

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define columns
columns_with_label = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
]
columns_without_label = columns_with_label[:-1]

# Load models
input_dim = len(scaler.feature_names_in_)
autoencoder = keras.Sequential([
    keras.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])
autoencoder.load_weights('models/autoencoder.weights.h5')

with open('models/iso_forest.pkl', 'rb') as f:
    iso_forest = pickle.load(f)

# Process file
if uploaded:
    df_test = pd.read_csv(uploaded, header=None)
    if df_test.shape[1] == len(columns_with_label):
        df_test.columns = columns_with_label
        X_test = df_test.drop(['label'], axis=1)
    elif df_test.shape[1] == len(columns_without_label):
        df_test.columns = columns_without_label
        X_test = df_test
    else:
        st.error("❌ Unexpected number of columns.")
        st.stop()
    st.success("✅ File uploaded & processed!")
else:
    st.info("Using default test data")
    df_test = pd.read_csv('data/kddcup.data_10_percent_corrected.csv', header=None)
    df_test.columns = columns_with_label
    X_test = df_test.drop(['label'], axis=1)

# Encode & scale
df_enc = pd.get_dummies(X_test, columns=["protocol_type", "service", "flag"])
df_enc = df_enc.reindex(columns=scaler.feature_names_in_, fill_value=0)
X_scaled_test = scaler.transform(df_enc)

# Predict
if model_choice == "Autoencoder (Deep Learning)":
    reconstructions = autoencoder.predict(X_scaled_test)
    loss = np.mean(np.square(X_scaled_test - reconstructions), axis=1)
    df_test['AnomalyScore'] = loss
    df_test['Anomaly'] = loss > np.percentile(loss, 95)
else:
    preds = iso_forest.decision_function(X_scaled_test)
    df_test['AnomalyScore'] = -preds
    df_test['Anomaly'] = iso_forest.predict(X_scaled_test) == -1

# 📊 Visuals (compact & unified design)
st.subheader("📊 Anomaly Score Histogram")
hist_counts, _ = np.histogram(df_test['AnomalyScore'], bins=30)
st.bar_chart(pd.DataFrame(hist_counts))

st.subheader("📈 Anomaly Score Trend")
sorted_scores = np.sort(df_test['AnomalyScore'])
st.line_chart(sorted_scores[:500])

st.subheader("📦 Moving Average (smooth trend)")
moving_avg = pd.Series(sorted_scores).rolling(window=50).mean()
st.line_chart(moving_avg[:500])

st.subheader("🔥 Top 10 Anomalies by Score")
st.dataframe(df_test.sort_values('AnomalyScore', ascending=False).head(10)[['AnomalyScore']])
