# 🛡️ Network Anomaly Detection using Autoencoder & Isolation Forest

This project uses **unsupervised machine learning** techniques — specifically a **deep learning autoencoder** and **Isolation Forest** — to detect unusual patterns or anomalies in network traffic data. Such anomalies may indicate potential security breaches, attacks, or unexpected system malfunctions.

Built with **Streamlit**, the dashboard allows you to interactively upload test data, choose the detection model, and visualize anomalies.

---

## 🚀 Problem Statement
> Using unsupervised learning techniques such as isolation forests or autoencoders to detect unusual patterns or anomalies in network traffic data, which could indicate potential security breaches or system malfunctions.

We use the **KDD Cup 1999** dataset, a classic benchmark for intrusion detection, containing features like protocol type, service, duration, and byte counts.

---

## 🧠 Models Used
- **Autoencoder (Deep Learning):**
  - Learns to reconstruct "normal" network traffic.
  - Large reconstruction error → anomaly.
- **Isolation Forest (Classical ML):**
  - Detects outliers by randomly partitioning feature space.
  - Points in small partitions → anomaly.

Both models are trained unsupervised (no need for attack labels).

---

## 📊 Dashboard Features
✅ Upload your own test CSV file (with or without labels).  
✅ Choose detection model (Autoencoder or Isolation Forest).  
✅ See:
- Reconstruction error histogram
- Top anomalies
- Boxplot & scatter plot for deeper insights
✅ Clean UI with thin side border style for aesthetics.


Here is the render link:
   https://anomaly-detection-in-network-traffic-1-s54a.onrender.com 