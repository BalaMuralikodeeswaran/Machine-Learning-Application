import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score  
import matplotlib.pyplot as plt
from io import BytesIO

# Helper functions
def plot_clusters(data, cluster_col):
  fig, ax = plt.subplots()
  ax.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data[cluster_col]) 
  return fig

def plot_elbow_curve(data):
  distortions = []
  for i in range(2, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(data)
    distortions.append(km.inertia_)
  
  fig, ax = plt.subplots()
  ax.plot(range(2, 11), distortions, marker='o')
  ax.set_xlabel('Number of clusters')
  ax.set_ylabel('Distortion')
  return fig

# Load data
uploaded_file = st.file_uploader("Choose a CSV file", type='csv') 
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file)
  
# Number of clusters
num_clusters = st.sidebar.number_input("Number of clusters", min_value=2, max_value=10, value=3)  

# Cluster data
numeric_data = data.select_dtypes(include=[np.number])
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(numeric_data)
data['cluster'] = kmeans.labels_ 

# Evaluation metrics
silhouette = silhouette_score(numeric_data, kmeans.labels_)
davies_bouldin = davies_bouldin_score(numeric_data, kmeans.labels_)

# Visualizations 
scatter_plot = plot_clusters(data, 'cluster')
st.pyplot(scatter_plot)

elbow_plot = plot_elbow_curve(numeric_data) 
st.pyplot(elbow_plot)

# Display metrics
st.write("Silhouette score:", silhouette)
st.write("Davies-Bouldin index:", davies_bouldin) 

st.write("""
The silhouette score of {} indicates a fairly good separation between clusters.
The Davies-Bouldin index of {} implies acceptable clustering.  
""".format(silhouette, davies_bouldin))