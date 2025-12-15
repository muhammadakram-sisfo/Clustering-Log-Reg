# ======================================================
# STREAMLIT APP: HCV DATA MINING
# Preprocessing + Clustering + Logistic Regression
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="HCV Data Mining",
    layout="wide"
)

st.title("📊 Data Mining Penyakit HCV")
st.write("Preprocessing • Clustering • Logistic Regression")

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    df = pd.read_csv("hcvdat0.csv")
    return df

df = load_data()

# ======================================================
# PREPROCESSING
# ======================================================
df.drop(columns=['Unnamed: 0'], inplace=True)

X = df.drop(columns=['Category'])
y = df['Category']

# Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Encode Sex
le_sex = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ======================================================
# SIDEBAR CONTROL
# ======================================================
st.sidebar.header("⚙️ Pengaturan")

n_clusters = st.sidebar.slider(
    "Jumlah Cluster (K-Means)",
    min_value=2,
    max_value=6,
    value=4
)

test_size = st.sidebar.slider(
    "Test Size (Logistic Regression)",
    min_value=0.1,
    max_value=0.4,
    value=0.2
)

# ======================================================
# CLUSTERING SECTION
# ======================================================
st.header("🔹 Clustering (K-Means)")

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = cluster_labels

# PCA untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clustering
fig_cluster, ax = plt.subplots()
scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels
)
ax.set_title("Visualisasi Clustering K-Means (PCA 2D)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")

st.pyplot(fig_cluster)

st.subheader("Distribusi Cluster")
st.write(df['Cluster'].value_counts())

# ======================================================
# LOGISTIC REGRESSION SECTION
# ======================================================
st.header("🔹 Logistic Regression (Klasifikasi)")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=test_size,
    random_state=42,
    stratify=y_encoded
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.write(f"### 🎯 Akurasi Model: **{accuracy:.2f}**")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax = plt.subplots()
im = ax.imshow(cm)

ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig_cm)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Data Mining HCV • K-Means • Logistic Regression • Streamlit")
