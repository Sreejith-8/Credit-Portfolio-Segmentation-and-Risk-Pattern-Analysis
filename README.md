# Credit Portfolio Segmentation and Risk Pattern Analysis

## Project Overview

Credit Portfolio Segmentation and Risk Pattern Analysis is a Machine Learning project focused on understanding borrower characteristics, identifying credit portfolio segments, analyzing historical default patterns, and detecting unusual borrower profiles.

The project uses approximately **1.35 million loan records** from the Lending Club dataset. It applies dimensionality reduction and clustering techniques to discover meaningful patterns in borrower financial and credit characteristics.

An interactive Streamlit dashboard is developed to present analytical findings, clustering comparisons, risk patterns, and business insights.

## Project Objectives

- Understand borrower financial and credit characteristics.
- Perform data preprocessing and exploratory data analysis.
- Create meaningful features for credit portfolio analysis.
- Apply dimensionality reduction techniques.
- Identify borrower segments using clustering algorithms.
- Analyze historical default patterns across borrower segments.
- Investigate unusual borrower profiles.
- Present findings through an interactive dashboard.

## Project Workflow

### 1. Data Collection and Understanding

Collected and examined the Lending Club dataset to understand its financial, credit, employment, and loan-related attributes.

### 2. Data Preprocessing

Prepared the dataset by handling missing values, duplicate records, and data inconsistencies to improve data quality.

### 3. Exploratory Data Analysis (EDA)

Explored borrower characteristics, loan distributions, financial patterns, and default distributions to understand the dataset.

### 4. Feature Engineering

Created additional features from existing attributes to support more meaningful borrower and credit portfolio analysis.

### 5. Dimensionality Reduction

Applied Principal Component Analysis (PCA) to reduce 24 selected features to 8 principal components while retaining approximately **92.21% of the variance**.

Used t-SNE to visualize borrower patterns in a two-dimensional space.

### 6. Clustering Analysis

Applied three unsupervised learning algorithms to identify borrower segments:

- **K-Means:** Identified six borrower segments.
- **Hierarchical Clustering:** Explored borrower groups using hierarchical relationships.
- **DBSCAN:** Identified dense borrower groups and unusual records.

### 7. Cluster Evaluation and Interpretation

Evaluated clustering performance using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

Analyzed borrower financial and credit characteristics to understand the differences between clusters.

### 8. Default Pattern Analysis

Compared historical default rates across identified borrower segments. The Default column was excluded from cluster formation and used only for post-clustering analysis.

### 9. Anomaly Detection

Investigated unusual borrower profiles identified as noise by DBSCAN and compared their financial characteristics with those of other borrowers.

### 10. Business Insights and Visualization

Developed an interactive Streamlit dashboard to present borrower segments, clustering performance, default patterns, anomaly detection results, and business recommendations.

## Dataset Information

- **Dataset:** Lending Club Loan Dataset
- **Total Records:** 1,347,681
- **Original Features:** 13
- **Selected and Engineered PCA Input Features:** 24
- **PCA Output Features:** 8
- **Variance Retained:** 92.21%

**Sampling Details:** PCA and K-Means were applied to the complete dataset. Hierarchical Clustering, DBSCAN, t-SNE, cluster evaluation, default pattern analysis, and anomaly detection were performed using the first 5,000 records. Therefore, results from these sample-based analyses may not represent the entire dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly
- Streamlit

## Dashboard Features

The interactive dashboard includes:

- Borrower Segment Distribution
- Clustering Algorithm Comparison
- Default Pattern Analysis
- Borrower Financial and Credit Profiles
- Anomaly Detection
- Clustering Evaluation Metrics
- t-SNE Borrower Pattern Visualization
- Business Insights and Recommendations

## Project Structure

The project repository contains the following files:

- **app.py:** Streamlit dashboard application.
- **kmeans_results.csv:** K-Means clustering results.
- **hierarchical_results.csv:** Hierarchical clustering results.
- **dbscan_results.csv:** DBSCAN clustering results.
- **tsne_results.csv:** Two-dimensional t-SNE visualization results.
- **README.md:** Project documentation.

## Key Outcomes

- Reduced 24 selected features to 8 principal components while retaining approximately 92.21% variance.
- Identified six borrower segments using K-Means.
- Compared three different clustering algorithms.
- Examined historical default-rate variations across borrower segments.
- Investigated unusual borrower profiles using DBSCAN.
- Developed an interactive dashboard to communicate analytical findings.

## Conclusion

This project demonstrates the application of data analysis and unsupervised Machine Learning techniques to credit portfolio segmentation and risk pattern analysis. By exploring borrower characteristics, identifying distinct segments, examining historical default patterns, and investigating unusual profiles, the project provides meaningful insights that can support better understanding of credit portfolios.
