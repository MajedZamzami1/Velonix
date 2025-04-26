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

# 4. KMeans clustering (k=2) directly on full 1536D embeddings
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(embeddings)
df['cluster'] = labels  # Assign clusters back to dataframe

# 5. (Optional) PCA 2D projection for visualization
# pca = PCA(n_components=2)
# embeddings_2d = pca.fit_transform(embeddings)

# plt.figure(figsize=(8, 6))
# colors = ['red', 'purple']
# for i in range(2):
#     cluster_points = embeddings_2d[labels == i]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f"Cluster {i}")
# plt.title("KMeans Clustering of Finance Embeddings (projected for visualization)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()
# plt.grid(True)
# plt.show()

# 6. Check what each cluster mostly represents (Finance or Non-Finance)
cluster_summary = df.groupby('cluster')['Finance'].value_counts()
#print("\nCluster Meaning Summary:")
#print(cluster_summary)

# 7. Predict a new sentence
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="To reach profitability within the first year, Velonix must secure a minimum of two on-premise enterprise clients, generating $300,000 - $430,000 in base license revenue. With typical clients opting for the support subscription as well, five deployments with full packages would yield from $695,000 to approximately $1,070,000 in total revenue based on the initial cost of developing the product."
)
new_embedding = np.array(response.data[0].embedding).reshape(1, -1)

# 8. Predict the cluster for the new sentence
predicted_cluster = kmeans.predict(new_embedding)
print(f"\nThe new sentence belongs to Cluster {predicted_cluster[0]}.")

# 9. Map cluster number to real-world meaning
# (based on majority labels seen earlier)
majority_labels = df.groupby('cluster')['Finance'].agg(lambda x: x.value_counts().idxmax())

real_label = majority_labels[predicted_cluster[0]]
print(f"The new sentence is classified as: {real_label} (based on cluster majority).")
