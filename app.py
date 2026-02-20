

import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data={
    "Annual_Income": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                      50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
                      90, 92, 94, 96, 98],
    "Spending_Score": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72,
                       45, 55, 65, 75, 85, 95, 35, 25, 15, 5,
                       99, 88, 77, 66, 55]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)
wcss = []

for i in range(1, 8):
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(df)
    wcss.append(model.inertia_)
plt.figure()
plt.plot(range(1, 8), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)


plt.figure()
plt.scatter(df["Annual_Income"], df["Spending_Score"], c=df["Cluster"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using KMeans")
plt.show()
income = int(input("Enter New Customer Income: "))
score = int(input("Enter New Customer Spending Score: "))

new_customer = [[income, score]]
cluster = kmeans.predict(new_customer)

print(f"\nNew Customer belongs to Cluster {cluster[0]}")


print("\nClustered Dataset:")
print(df)