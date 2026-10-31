import streamlit as st
import pandas as pd
import plotly.express as px
kmeans_df = pd.read_csv('kmeans_results.csv')

hierarchical_df = pd.read_csv('hierarchical_results.csv')

dbscan_df = pd.read_csv('dbscan_results.csv')
tsne_df = pd.read_csv('tsne_results.csv')
st.set_page_config(
    page_title="Credit Portfolio Segmentation and Risk Pattern Analysis",
    layout="wide"
)

st.title("Credit Portfolio Segmentation and Risk Pattern Analysis")

st.write(
    "An interactive dashboard to explore borrower segments, "
    "clustering comparisons, default patterns, and business insights."
)
st.header("Borrower Segment Distribution")

cluster_counts = kmeans_df['Cluster'].value_counts().sort_index()

fig = px.bar(
    x=cluster_counts.index,
    y=cluster_counts.values,
    labels={
        'x': 'Borrower Cluster',
        'y': 'Number of Borrowers'
    },
    title='K-Means Borrower Distribution'
)

st.plotly_chart(fig, use_container_width=True)
st.header("Clustering Algorithm Comparison")

algorithm_counts = pd.DataFrame({
    'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN'],
    'Number of Clusters': [
        kmeans_df['Cluster'].nunique(),
        hierarchical_df['Cluster'].nunique(),
        dbscan_df[dbscan_df['Cluster'] != -1]['Cluster'].nunique()
    ]
})

fig = px.bar(
    algorithm_counts,
    x='Algorithm',
    y='Number of Clusters',
    title='Number of Clusters Identified by Each Algorithm',
    text='Number of Clusters'
)

st.plotly_chart(fig, use_container_width=True)
st.header("Default Pattern Analysis")

default_comparison = pd.DataFrame({
    'K-Means': kmeans_df.groupby('Cluster')['Default'].mean() * 100,
    'Hierarchical': hierarchical_df.groupby('Cluster')['Default'].mean() * 100,
    'DBSCAN': dbscan_df[dbscan_df['Cluster'] != -1]
        .groupby('Cluster')['Default'].mean() * 100
})

default_comparison = default_comparison.reset_index().melt(
    id_vars='Cluster',
    var_name='Algorithm',
    value_name='Default Rate'
)

fig = px.bar(
    default_comparison,
    x='Cluster',
    y='Default Rate',
    color='Algorithm',
    barmode='group',
    title='Default Rate Comparison Across Clustering Algorithms'
)

st.plotly_chart(fig, use_container_width=True)
st.header("Borrower Financial and Credit Profiles")

algorithm = st.selectbox(
    "Choose Clustering Algorithm",
    ["K-Means", "Hierarchical", "DBSCAN"]
)

if algorithm == "K-Means":
    selected_df = kmeans_df
elif algorithm == "Hierarchical":
    selected_df = hierarchical_df
else:
    selected_df = dbscan_df[dbscan_df['Cluster'] != -1]

features = ['revenue', 'dti_n', 'loan_amnt', 'fico_n', 'emp_length']

profile = selected_df.groupby('Cluster')[features].mean()

scaled_profile = (
    profile - profile.mean()
) / profile.std()

fig = px.imshow(
    scaled_profile,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    title=f'{algorithm} Borrower Profile Heatmap',
    labels=dict(
        x="Financial Features",
        y="Borrower Cluster",
        color="Relative Value"
    )
)

st.plotly_chart(fig, use_container_width=True)
st.header("Anomaly Detection")

anomaly_df = dbscan_df.copy()

anomaly_df['Record Type'] = anomaly_df['Cluster'].apply(
    lambda x: 'Noise' if x == -1 else 'Normal'
)

fig = px.scatter(
    anomaly_df,
    x='revenue',
    y='loan_amnt',
    color='Record Type',
    title='DBSCAN Noise vs Normal Borrowers',
    labels={
        'revenue': 'Revenue',
        'loan_amnt': 'Loan Amount'
    },
    opacity=0.6
)

st.plotly_chart(fig, use_container_width=True)
st.header("Business Insights and Recommendations")

st.subheader("Key Findings")

st.markdown("""
- K-Means identified 6 borrower segments.
- Hierarchical clustering identified 6 borrower segments.
- DBSCAN identified 3 borrower clusters and 231 noise records.
- K-Means Cluster 5 showed the highest default rate (25.91%).
- Hierarchical Cluster 5 showed the highest default rate (23.71%).
- DBSCAN Cluster 2 showed the highest default rate (22.71%).
- DBSCAN noise borrowers had higher average revenue and FICO scores
  than normal borrowers in our sample.
""")

st.subheader("Business Recommendations")

st.markdown("""
1. Monitor borrower segments with higher historical default rates.
2. Review repayment capacity before approving high-loan applications.
3. Investigate unusual borrower profiles separately.
4. Use credit score, revenue and DTI together when assessing borrowers.
""")
st.header("Clustering Evaluation Metrics")

metrics_df = pd.DataFrame({
    'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN'],
    'Silhouette Score': [0.131749, 0.139291, 0.164227],
    'Davies-Bouldin Index': [1.8051, 1.9369, 2.0948],
    'Calinski-Harabasz Index': [696.81, 553.05, 607.86]
})

st.dataframe(metrics_df, use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    fig = px.bar(
        metrics_df,
        x='Algorithm',
        y='Silhouette Score',
        title='Silhouette Score'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        metrics_df,
        x='Algorithm',
        y='Davies-Bouldin Index',
        title='Davies-Bouldin Index'
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = px.bar(
        metrics_df,
        x='Algorithm',
        y='Calinski-Harabasz Index',
        title='Calinski-Harabasz Index'
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "DBSCAN evaluation metrics exclude noise records. "
    "The metrics were calculated using the 5,000-record sample."
)
st.header("Borrower Pattern Visualization (t-SNE)")

fig = px.scatter(
    tsne_df,
    x='TSNE1',
    y='TSNE2',
    color='Cluster',
    title='Borrower Patterns Using t-SNE'
)

st.plotly_chart(fig, use_container_width=True)