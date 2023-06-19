import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

model = pickle.load(open('output_marketing_fruit.sav', 'rb'))

df=pd.read_excel("output_cluster_market.xlsx")
features = ['Recency', 'Fruit']
X = df[features]

st.title('Supermarket Fruit Sales')

n_clust = st.slider("Select Number of Clusters", min_value=1, max_value=10, value=4)

model.n_clusters = n_clust
clusters = model.fit_predict(X)

fig, ax = plt.subplots()
scatter = ax.scatter(X['Recency'], X['Fruit'], c=clusters, cmap='viridis')
ax.set_xlabel('Recency')
ax.set_ylabel('Fruit')


legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

st.pyplot(fig)