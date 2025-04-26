import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load .env variables and OpenAI client
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# 1. Load your CSV
df = pd.read_csv('financeData.csv')

# 2. Parse embeddings
df['embedding'] = df['embedding'].apply(ast.literal_eval)
embeddings = np.vstack(df['embedding'].values)

# 3. Convert labels ("Yes" -> 1, "No" -> 0)
labels = df['Finance'].apply(lambda x: 1 if x == 'Yes' else 0).values

# 4. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# 5. Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Predict a new sentence
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="During the Napoleonic era, Spain became a French puppet state. Concurrent with, and following, the Napoleonic period the Spanish American wars of independence resulted in the loss of most of Spain's territory in the Americas in the 1820s. During the re-establishment of the Bourbon rule in Spain, constitutional monarchy was introduced in 1813."
)
new_embedding = np.array(response.data[0].embedding).reshape(1, -1)

new_pred = model.predict(new_embedding)

if new_pred[0] == 1:
    print("The new sentence is classified as FINANCE.")
else:
    print("The new sentence is classified as NON-FINANCE.")
