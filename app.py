
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Customer Insights Dashboard")

st.markdown(
    "Explore customer behaviour and discover meaningful "
    "customer segments using Machine Learning."
)

st.divider()


# --------------------------------
# 2. Upload Dataset
# --------------------------------

st.sidebar.header("Project Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Mall Customers CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👈 Upload your Mall Customers CSV file from the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

missing_columns = [col for col in features if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

if df[features].isnull().any().any():
    st.error("The selected features contain missing values. Please clean the dataset.")
    st.stop()

# --------------------------------
# 3. Dataset Overview
# --------------------------------

st.header("📈 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Customers",
        value=len(df)
    )

with col2:
    st.metric(
        label="Total Features",
        value=len(features)
    )

with col3:
    st.metric(
        label="Missing Values",
        value=int(df[features].isnull().sum().sum())
    )

# --------------------------------
# 4. Sidebar Model Settings
# --------------------------------

st.sidebar.subheader("Clustering Settings")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["K-Means", "Hierarchical Clustering"]
)

k = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=6
)

metric = "euclidean"
linkage_method = "ward"

if algorithm == "Hierarchical Clustering":

    linkage_method = st.sidebar.selectbox(
        "Select Linkage",
        ["ward", "average", "complete"]
    )

    if linkage_method == "ward":
        metric = "euclidean"
        st.sidebar.caption("Ward linkage uses Euclidean distance.")

    else:
        metric = st.sidebar.selectbox(
            "Select Distance Metric",
            ["euclidean", "manhattan", "cosine"]
        )

# --------------------------------
# 5. Feature Scaling
# --------------------------------

X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------
# 6. Apply Clustering
# --------------------------------

if algorithm == "K-Means":

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)
    evaluation_metric = "euclidean"

else:

    model = AgglomerativeClustering(
        n_clusters=k,
        metric=metric,
        linkage=linkage_method
    )

    labels = model.fit_predict(X_scaled)
    evaluation_metric = metric

result_df = df.copy()
result_df["Cluster"] = labels

# --------------------------------
# 7. Model Evaluation
# --------------------------------

silhouette = silhouette_score(
    X_scaled,
    labels,
    metric=evaluation_metric
)

db_index = davies_bouldin_score(X_scaled, labels)
ch_score = calinski_harabasz_score(X_scaled, labels)


# --------------------------------
# Dashboard Navigation
# --------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "👥 Customer Analysis",
    "📈 Model Evaluation",
    "🔍 EDA"
])

st.divider()
st.header("Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("Silhouette Score", f"{silhouette:.4f}")
col2.metric("Davies-Bouldin Index", f"{db_index:.4f}")
col3.metric("Calinski-Harabasz Score", f"{ch_score:.2f}")

st.caption(
    "Silhouette uses the selected model distance. "
    "Davies-Bouldin and Calinski-Harabasz use Euclidean-based definitions."
)


# --------------------------------
# TAB 1: Dashboard
# --------------------------------

with tab1:

    st.subheader("Customer Segmentation")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=result_df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="viridis",
        s=90,
        ax=ax
    )

    ax.set_title(f"{algorithm} (K={k})")
    ax.grid(alpha=0.3)
    ax.legend(title="Customer Cluster")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Customer Distribution")

    cluster_counts = (
        result_df["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(cluster_counts)


# --------------------------------
# TAB 2: Customer Analysis
# --------------------------------

with tab2:

    st.subheader("Customer Segment Profiles")

    cluster_profile = (
        result_df.groupby("Cluster")[features]
        .mean()
        .round(2)
    )

    cluster_profile.insert(
        0,
        "Customer Count",
        cluster_counts
    )

    st.dataframe(
        cluster_profile,
        use_container_width=True
    )

    st.subheader("Customer Records")

    st.dataframe(
        result_df,
        use_container_width=True
    )

    st.download_button(
        label="Download Customer Results",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="customer_segmentation_results.csv",
        mime="text/csv"
    )

# --------------------------------
# TAB 3: Model Evaluation
# --------------------------------

with tab3:

    st.subheader("Clustering Performance")

    evaluation_df = pd.DataFrame({
        "Metric": [
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Score"
        ],
        "Value": [
            silhouette,
            db_index,
            ch_score
        ]
    })

    st.dataframe(
        evaluation_df,
        use_container_width=True
    )

    st.info(
        f"Selected Algorithm: {algorithm}\n\n"
        f"Number of Clusters: {k}\n\n"
        f"Distance Metric: {evaluation_metric}"
    )

    st.divider()

    st.subheader("🏆 Model Performance Insights")

    st.write(
        "**Best observed configuration:** "
        "Hierarchical Clustering with Cosine distance, "
        "Average linkage and K=6."
    )

    st.markdown("""
    - **K-Means:** Produced reasonably well-separated customer groups.
    - **Hierarchical Euclidean:** Achieved a result comparable to K-Means.
    - **Hierarchical Cosine:** Achieved the highest Silhouette Score
      among the tested configurations.
    """)

    st.divider()

    st.subheader("📝 Project Conclusion")

    st.write(
        "Hierarchical Clustering using Cosine distance and "
        "Average linkage achieved the highest Silhouette Score "
        "among the tested configurations. The experiments show "
        "how distance metrics and linkage methods influence "
        "customer segmentation results."
    )

st.success("Customer segmentation completed successfully!")


# --------------------------------
# TAB 4: Exploratory Data Analysis
# --------------------------------

with tab4:

    st.subheader("Exploratory Data Analysis")

    st.write("Understand the distribution of customer features.")

    # Feature Distributions

    for feature in features:

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.histplot(
            data=df,
            x=feature,
            kde=True,
            bins=15,
            ax=ax
        )

        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Number of Customers")

        st.pyplot(fig)
        plt.close(fig)

    # Correlation Heatmap

    st.subheader("Feature Correlation")

    correlation = df[features].corr()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Customer Feature Correlation")

    st.pyplot(fig)
    plt.close(fig)
