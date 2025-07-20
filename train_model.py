# train_model.py
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow import keras
from tensorflow.keras import layers

print("[INFO] Loading data...")
df = pd.read_csv('data/kddcup.data_10_percent_corrected.csv', header=None)

# Define column names
columns = [
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
df.columns = columns

# Encode categorical
df_enc = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

# Remove label
X = df_enc.drop(['label'], axis=1)

print("[INFO] Preprocessing...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("[INFO] Training autoencoder...")
input_dim = X_scaled.shape[1]
autoencoder = keras.Sequential([
    keras.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=256, validation_split=0.1)
autoencoder.save_weights('models/autoencoder.weights.h5')

print("[INFO] Training Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_scaled)
with open('models/iso_forest.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)

print("[✅] Training complete! Both models saved.")
