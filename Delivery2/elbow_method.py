import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load the dataset from a CSV file
df = pd.read_csv('dataset_clean.csv')

# Convert datetime columns and extract useful numerical features
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['year'] = df['Order Date'].dt.year


# Drop the original datetime column after extraction
df.drop('Order Date', axis=1, inplace=True)

# Optionally, drop other non-relevant or non-numeric columns
df = df.select_dtypes(include=[np.number])


# Scale the numeric data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

from sklearn.impute import SimpleImputer

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Apply the imputer to your dataset
# Assuming 'scaled_df' from previous steps is the DataFrame you are working with
scaled_df_imputed = imputer.fit_transform(scaled_df)

# The output will be a numpy array, you can convert it back to DataFrame if needed
scaled_df_imputed = pd.DataFrame(scaled_df_imputed, columns=df.columns)


# Drop all rows with any NaN values
df.dropna(inplace=True)

# Proceed with scaling again since the dataset has changed
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)


# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)  # make sure to use the cleaned dataset (e.g., 'scaled_df' or 'scaled_df_imputed')
    sse.append(kmeans.inertia_)

# Plotting the SSE values to find the elbow
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances (SSE)')
plt.show()

# Choose the number of clusters and apply K-Means
k = 5  # Replace with the number chosen from the elbow method
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_df)  # use the appropriate dataset

# Add the cluster labels to your DataFrame
df['Cluster'] = clusters
print(df.head())

# Continue with any further analysis or visualization as needed

from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import dendrogram, linkage

imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df.select_dtypes(include=[np.number]))

# Scale the data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_imputed)

# Generate the linkage matrix using Ward's method
Z = linkage(scaled_df, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 7))
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster size')
plt.ylabel('Distance')
plt.show()
