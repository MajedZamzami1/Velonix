import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import ast
from openai import OpenAI
from dotenv import load_dotenv
import os
from sklearn.decomposition import PCA

# 1. Load environment variables and OpenAI client
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# 2. Load the CSV file
df = pd.read_csv('financeData.csv')

# 3. Parse embeddings (convert string to list)
df['embedding'] = df['embedding'].apply(ast.literal_eval)
embeddings = np.vstack(df['embedding'].values)

# 4. Separate Finance and Non-Finance embeddings
finance_embeddings = np.vstack(df[df['Finance'] == 'Yes']['embedding'].values)
non_finance_embeddings = np.vstack(df[df['Finance'] == 'No']['embedding'].values)

# 5. Calculate smarter initial centroids
finance_centroid = np.mean(finance_embeddings, axis=0)
non_finance_centroid = np.mean(non_finance_embeddings, axis=0)

# 6. Initialize KMeans with custom centers
custom_centers = np.vstack([finance_centroid, non_finance_centroid])
kmeans = KMeans(n_clusters=2, init=custom_centers, n_init=1, random_state=0)
kmeans.fit(embeddings)

# ✅ Add this line to assign the cluster labels
df['cluster'] = kmeans.labels_

# 7. Now check cluster meaning
majority_labels = df.groupby('cluster')['Finance'].agg(lambda x: x.value_counts().idxmax())


print("\nCluster Meaning Summary:")
print(majority_labels)

# 8. Predict a new sentence
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Feasibility &  Functionality  Validate and test Velonix’s seamless integration with existing  systems and its ability to prevent unauthorized data access.  Stakeholder Feedback  Gather feedback from industry partners to improve performance,  usability, and compliance features. 2. System Architecture Overview      2.1 Generic Architecture    Here are the three architecture diagrams showing different placements of the Data Control  Filtering in the RAG pipeline:    Architecture 1: Data Control Filtering Before Prompt  Data is filtered based on user permissions even before a query is made. This approach ensures  users only interact with datasets they are allowed to query. In this architecture, queries run on a  smaller dataset, reducing computation time and improving response speed. This is also the  most secure and effective in reducing data leakages however it is not flexible and if the filter is  too strict, the LLM may not retrieve enough context, leading to lower-quality responses ."
)
new_embedding = np.array(response.data[0].embedding).reshape(1, -1)

predicted_cluster = kmeans.predict(new_embedding)
print(f"\nThe new sentence belongs to Cluster {predicted_cluster[0]}.")

real_label = majority_labels[predicted_cluster[0]]
print(f"The new sentence is classified as: {real_label} (based on cluster majority).")

# 9. (Optional) 2D visualization
# pca = PCA(n_components=2)
# embeddings_2d = pca.fit_transform(embeddings)

# plt.figure(figsize=(8, 6))
# colors = ['red', 'purple']
# for i in range(2):
#     cluster_points = embeddings_2d[kmeans.labels_ == i]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f"Cluster {i}")
# plt.title("KMeans Clustering of Finance Embeddings (Smart Init)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()
# plt.grid(True)
# plt.show()
