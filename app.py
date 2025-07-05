import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Credit Card Customer Segmentation")

# --- Load external CSS ---
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# --- Modern Header ---
st.markdown("""
<div class="modern-header">
    <h1>Credit Card Customer Segmentation</h1>
    <p>Discover customer groups and predict your segment instantly</p>
</div>
""", unsafe_allow_html=True)

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv("CC GENERAL.csv").fillna(method="ffill")
    return df

df = load_data()
data = df.drop(columns=["CUST_ID"])

# --- Train model on all features ---
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaled)
pca = PCA(2).fit(scaled)

# --- Minimal user input ---
st.sidebar.header("User Input")
user_features = ["BALANCE", "PURCHASES", "CREDIT_LIMIT", "TENURE", "PAYMENTS"]  # Added TENURE and PAYMENTS
inputs = {
    col: st.sidebar.number_input(
        f"{col.replace('_', ' ').title()}",
        float(data[col].min()), float(data[col].max()), float(data[col].min())
    )
    for col in user_features
}
u_df = pd.DataFrame([inputs])

# Fill missing columns with mean for prediction
for col in data.columns:
    if col not in u_df.columns:
        u_df[col] = data[col].mean()
u_df = u_df[data.columns]

# --- Only show results if all user inputs are provided ---
if all(inputs[col] != float(data[col].min()) for col in user_features):
    # --- Predict cluster ---
    u_scaled = scaler.transform(u_df)
    cluster = kmeans.predict(u_scaled)[0]

    # --- Show prediction badge ---
    st.markdown(f"""
    <div class="prediction-badge">
        <span class="badge-main">
            Predicted Cluster: {cluster}
        </span>
        <span class="badge-sub">
            (Based on your input)
        </span>
    </div>
    """, unsafe_allow_html=True)

    # --- Show user input summary ---
    st.markdown("#### Your Input Summary")
    st.table(u_df[user_features].T.rename(columns={0: "Value"}))

    # --- Visualize clusters with PCA ---
    coords = pca.transform(scaled)
    u_coord = pca.transform(u_scaled)
    centroids = pca.transform(kmeans.cluster_centers_)

    st.markdown("#### Cluster Visualization (PCA Projection)")
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=kmeans.labels_, cmap="tab10", alpha=0.35, label="Cluster Members"
    )
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        c="black", s=180, marker="X", label="Centroids"
    )
    ax.scatter(
        u_coord[0, 0], u_coord[0, 1],
        c="#fbbf24", s=180, marker="*", label="Your Data"
    )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("K-Means Clusters with Centroids")
    ax.legend()
    st.pyplot(fig)

    # --- Radar/Spider Chart for cluster profiles ---
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=data.columns)
    categories = user_features  # or any features you want to show
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for i, row in cluster_centers.iterrows():
        values = row[categories].tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.legend()
    st.pyplot(fig)

    # --- Modern Demographics Card (from HTML) ---
    try:
        with open("demographics.html") as f:
            demographics_html = f.read().format(
                total_customers=f"{len(df):,}",
                avg_balance=f"{df['BALANCE'].mean():,.2f}",
                median_credit_limit=f"{df['CREDIT_LIMIT'].median():,.2f}",
                full_payments=f"{df['PRC_FULL_PAYMENT'].mean() * 100:.1f}"
            )
        st.markdown(demographics_html, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("demographics.html not found. Please add it for the demographics card.")
else:
    st.info("Please enter all user inputs to see the cluster prediction and visualizations.")

