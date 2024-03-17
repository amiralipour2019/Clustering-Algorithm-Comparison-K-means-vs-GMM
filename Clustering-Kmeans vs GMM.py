
pip install umap-learn scikit-learn matplotlib pandas

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP

from google.colab import drive
drive.mount('/content/drive')

# Loading the dataset
dataset_path = '/content/drive/s3.txt'
s3_data = pd.read_csv(dataset_path, sep="\s+", header=None)  # Assuming whitespace separator and no header

# Displaying the first few rows of the dataset
s3_data.head()
s3_data.shape

# Check for missing values
missing_values=s3_data.isnull().sum()
missing_values

# Standardizing the data
scaler=StandardScaler()
s3_data_scaled=scaler.fit_transform(s3_data)

# Convert the scaled data back to a DataFrame for easier handling in outlier detection
s3_df_scaled=pd.DataFrame(s3_data_scaled,columns=["0","1"])
s3_df_scaled

# Overview of potential outliers by describing the data
s3_df_summary=s3_df_scaled.describe()
s3_df_summary

n_components_range=range(1,21)
bics=[]
aics=[]

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(s3_df_scaled)
    bics.append(gmm.bic(s3_df_scaled))
    aics.append(gmm.aic(s3_df_scaled))

# Plotting the BIC and AIC values for different numbers of components
plt.figure(figsize=(14, 6))

plt.plot(n_components_range, bics, label='BIC', marker='o')
plt.plot(n_components_range, aics, label='AIC', marker='o')
plt.legend(loc='best')
plt.xlabel('Number of Components')
plt.ylabel('Criteria Value')
#plt.title('BIC and AIC for Different Number of Components')
plt.xticks(n_components_range)
plt.grid(True)

plt.show()

# Apply Gaussian Mixture Model with optimal number of components=8
gmm_model=GaussianMixture(n_components=15,random_state=56)
gmm_clusters=gmm_model.fit_predict(s3_df_scaled)

gmm_clusters

# Visualizing the clusters
plt.figure(figsize=(16,8))

# Scatter plot of the data points, colored by their cluster assignment
plt.scatter(s3_data_scaled[:,0],s3_data_scaled[:,1],c=gmm_clusters,cmap="viridis",marker=".")
#plt.title('Clusters Visualization with Gaussian Mixture Model')
plt.xlabel('Standardized X')
plt.ylabel('Standardized Y')
plt.colorbar(label='Cluster Label')
plt.show()

# Attaching the cluster labels to the original data
s3_df_with_clusters=s3_data.copy()
s3_df_with_clusters['Clusters']=gmm_clusters

# Previewing the data with cluster labels
s3_df_with_clusters.head()

"""**S Sets Analysis**"""



# Paths to the uploaded S Sets files
S3_path="/content/drive/s3.txt"
s4_path="/content/drive/s4.txt"

s3_data = pd.read_csv(S3_path,sep="\s+", header=None, names=["X1", "X2"])
s4_data = pd.read_csv(s4_path,sep="\s+", header=None, names=["X1", "X2"])

print(s3_data.shape)
print(s3_data.head())

print(s4_data.shape)
print(s4_data.head())

# Adjust these font sizes as needed
title_fontsize = 40
label_fontsize = 35
ticks_fontsize = 28
tick_length = 20
tick_width = 7

# Visualize the Birch sets
plt.figure(figsize=(20,10))

# S3
plt.subplot(1,2,1)
plt.scatter(s3_data["X1"],s3_data["X2"],s=0.5,color="red")
plt.title('S3 Dataset ',fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)


# S4
plt.subplot(1,2,2)
plt.scatter(s4_data["X1"],s4_data["X2"],s=0.5,color="black")
plt.title('S4 Dataset',fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)


plt.tight_layout()
plt.show()

# Applying K-means clustering on S sets

#S3
kmeans_model_s3=KMeans(n_clusters=15,random_state=561)
kmeans_clusters_s3=kmeans_model_s3.fit_predict(s3_data)

#S4
kmeans_model_s4=KMeans(n_clusters=15,random_state=561)
kmeans_clusters_s4=kmeans_model_s4.fit_predict(s4_data)

unique_labels_km_s3=np.unique(kmeans_clusters_s3)
unique_labels_km_s4=np.unique(kmeans_clusters_s4)

print(unique_labels_km_s3)
print(unique_labels_km_s4)

# Visualizing K-means Clustering of S sets
plt.figure(figsize=(20,10))

# S3
plt.subplot(1,2,1)
plt.scatter(s3_data["X1"], s3_data["X2"], c=kmeans_clusters_s3, cmap="viridis", marker=".")
plt.title("K-means Clustering of S3", fontsize=title_fontsize)
plt.xlabel("X1", fontsize=label_fontsize)
plt.ylabel("X2", fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize, length=tick_length, width=tick_width)

# S4
plt.subplot(1,2,2)
plt.scatter(s4_data["X1"], s4_data["X2"], c=kmeans_clusters_s4, cmap="viridis", marker=".")
plt.title("K-means Clustering of S4", fontsize=title_fontsize)
plt.xlabel("X1", fontsize=label_fontsize)
plt.ylabel("X2", fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize, length=tick_length, width=tick_width)

# Adjusting the color bar size
cbar = plt.colorbar(label='Cluster Label')
cbar.ax.tick_params(labelsize=ticks_fontsize)

plt.tight_layout()
plt.show()



# Applying Gaussian Mixture Model clustering on S sets

#S3
gmm_model_s3=GaussianMixture(n_components=15,random_state=561)
gmm_clusters_s3=gmm_model_s3.fit_predict(s3_data)

#S4
gmm_model_s4=GaussianMixture(n_components=15,random_state=561)
gmm_clusters_s4=gmm_model_s4.fit_predict(s4_data)

gmm_clusters_s3

# Visualizing GMM Clustering of S sets
plt.figure(figsize=(20,10))

# S3
plt.subplot(1,2,1)
plt.scatter(s3_data["X1"],s3_data["X2"], c=gmm_clusters_s3, cmap="viridis",marker=".")
plt.title("GMM Clustering of S3", fontsize=title_fontsize)
plt.xlabel("X1", fontsize=label_fontsize)
plt.ylabel("X2", fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize, length=tick_length, width=tick_width)

# S4
plt.subplot(1,2,2)
plt.scatter(s4_data["X1"],s4_data["X2"], c=gmm_clusters_s4, cmap="viridis",marker=".")
plt.title("GMM Clustering of S4", fontsize=title_fontsize)
plt.xlabel("X1", fontsize=label_fontsize)
plt.ylabel("X2", fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize, length=tick_length, width=tick_width)

# Add colorbar and adjust its size
cbar = plt.colorbar(label='Cluster Label')
cbar.ax.tick_params(labelsize=ticks_fontsize)

plt.tight_layout()
plt.show()





# Load the ground trouth S3 and S4
s3_ground_trouth_path="/content/s3-label.txt"
s4_ground_trouth_path="/content/drive/s4-label.txt"



s3_ground_truth_labels=pd.read_csv(s3_ground_trouth_path,sep="\+",header=None)
s4_ground_truth_labels=pd.read_csv(s4_ground_trouth_path,sep="\+",header=None)

print(s3_ground_truth_labels.shape)
print(s4_ground_truth_labels.head())

# Extract the labels of S Sets

#S3
s3_ground_truth_labels_final=s3_ground_truth_labels.iloc[5:].reset_index(drop=True)
s4_ground_truth_labels_final=s4_ground_truth_labels.iloc[5:].reset_index(drop=True)

print(s3_ground_truth_labels_final.shape)
print(s3_ground_truth_labels_final.head())

print(s4_ground_truth_labels_final.shape)
print(s4_ground_truth_labels_final.head())

# Fix the labels format of S Sets

#S3
s3_ground_truth_labels_final=s3_ground_truth_labels_final[0].astype(int)

#S4
s4_ground_truth_labels_final=s4_ground_truth_labels_final[0].astype(int)

print(s3_ground_truth_labels_final)

unique_true_labels_s3=np.unique(s3_ground_truth_labels_final)

print(unique_true_labels_s3)

print(s4_ground_truth_labels_final)

# Calcualte ARI and NMI Metrics for S Sets

############# S3
#Kmeans
ari_kmeans_s3=adjusted_rand_score(s3_ground_truth_labels_final,kmeans_clusters_s3)
nmi_kmeans_s3=normalized_mutual_info_score(s3_ground_truth_labels_final,kmeans_clusters_s3)

#GMM
ari_gmm_s3=adjusted_rand_score(s3_ground_truth_labels_final,gmm_clusters_s3)
nmi_gmm_s3=normalized_mutual_info_score(s3_ground_truth_labels_final,gmm_clusters_s3)

############# S4
#Kmeans
ari_kmeans_s4=adjusted_rand_score(s4_ground_truth_labels_final,kmeans_clusters_s4)
nmi_kmeans_s4=normalized_mutual_info_score(s4_ground_truth_labels_final,kmeans_clusters_s4)

#GMM
ari_gmm_s4=adjusted_rand_score(s4_ground_truth_labels_final,gmm_clusters_s4)
nmi_gmm_s4=normalized_mutual_info_score(s4_ground_truth_labels_final,gmm_clusters_s4)

#S3
print(ari_kmeans_s3,nmi_kmeans_s3)
print(ari_gmm_s3,nmi_gmm_s3)

# S4
print(ari_kmeans_s4,nmi_kmeans_s4)
print(ari_gmm_s4,nmi_gmm_s4)





"""**A-Sets Analysis**"""

#Load A3 dataset
a3_file_path="/content/a3.txt"
a3_data=pd.read_csv(a3_file_path,sep="\s+",header=None)

print(a3_data.shape)
print(a3_data.head())

plt.subplot(1,2,1)
plt.scatter(s3_data["X1"],s3_data["X2"],s=0.5,color="red")
plt.title('S3 Dataset ',fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

#  Visualize the dataset to understand its structure better
plt.figure(figsize=(20,10))
plt.scatter(a3_data[0],a3_data[1],s=10,color="red")
#plt.title('A3 Dataset Visualization')
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

plt.tight_layout()
plt.show()

# Basic statistics of A3 set
a3_data.describe()

# Applying K-means clustering on A3 set
kmeans_model_a3=KMeans(n_clusters=50,random_state=56)
kmeans_clusters_a3=kmeans_model_a3.fit_predict(a3_data)

# Applying Gaussian Mixture Model clustering
gmm_model_a3=GaussianMixture(n_components=50,random_state=561)
gmm_clusters_a3=gmm_model_a3.fit_predict(a3_data)

gmm_clusters_a3

# Visualizing K-means Clustering of A3-Set
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.scatter(a3_data[0],a3_data[1],c=kmeans_clusters_a3,cmap="viridis",marker=".")
plt.title("Kmeans Clustering of A3 set",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

# Visualizing GMM Clustering of A3-Set
#plt.figure(figsize=(16,8))
plt.subplot(1,2,2)
plt.scatter(a3_data[0],a3_data[1],c=gmm_clusters_a3,cmap="viridis",marker=".")
plt.title("GMM Clustering of A3 set",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

plt.tight_layout()
plt.show()

a3_ground_trouth_path="/content/a3-Ground truth partitions.txt"
a3_ground_truth_labels=pd.read_csv(a3_ground_trouth_path,sep="\+",header=None)

print(a3_ground_truth_labels.shape)
print(a3_ground_truth_labels.head())

# Extract the labels of A3 Set
a3_ground_truth_labels_final=a3_ground_truth_labels.iloc[4:].reset_index(drop=True)

print(a3_ground_truth_labels_final.shape)
print(a3_ground_truth_labels_final.head())

# Fix the labels format of A3 Set
a3_ground_truth_labels_final=a3_ground_truth_labels_final[0].astype(int)

print(a3_ground_truth_labels_final)

# Calcualte ARI and NMI Metrics for A3 Set
#Kmeans
ari_kmeans_a3=adjusted_rand_score(a3_ground_truth_labels_final,kmeans_clusters_a3)
nmi_kmeans_a3=normalized_mutual_info_score(a3_ground_truth_labels_final,kmeans_clusters_a3)

#GMM
ari_gmm_a3=adjusted_rand_score(a3_ground_truth_labels_final,gmm_clusters_a3)
nmi_gmm_a3=normalized_mutual_info_score(a3_ground_truth_labels_final,gmm_clusters_a3)

print(ari_kmeans_a3,nmi_kmeans_a3)

print(ari_gmm_a3,nmi_gmm_a3)











"""**Birch Sets Clustering Analysis**"""

# Paths to the uploaded files
birch1_path="/content/drive/Birch Sets/birch1.txt"
birch2_path="/content/drive/Birch Sets/birch2.txt"
birch3_path="/content/drive/Birch Sets/birch3.txt"

birch1_data = pd.read_csv(birch1_path,sep="\s+", header=None, names=["X1", "X2"])
birch2_data = pd.read_csv(birch2_path,sep="\s+", header=None, names=["X1", "X2"])
birch3_data = pd.read_csv(birch3_path,sep="\s+", header=None, names=["X1", "X2"])

print(birch1_data.shape)
print(birch1_data.head())

print(birch2_data.shape)
print(birch2_data.head())

print(birch3_data.shape)
print(birch3_data.head())

# Visualize the Birch sets
plt.figure(figsize=(20,10))

# Birch1
plt.subplot(1,3,1)
plt.scatter(birch1_data["X1"],birch1_data["X2"],s=0.5,color="red")
plt.title('Birch-1 ',fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

# Birch 2
plt.subplot(1,3,2)
plt.scatter(birch2_data["X1"],birch2_data["X2"],s=0.5,color="black")
plt.title('Birch-2 ',fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

# Birch 3
plt.subplot(1,3,3)
plt.scatter(birch3_data["X1"],birch3_data["X2"],s=0.5)
plt.title('Birch-3 ',fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

plt.tight_layout()
plt.show()

# Applying K-means clustering on Birch sets

#Birch1
kmeans_model_birch1=KMeans(n_clusters=50,random_state=561)
kmeans_clusters_birch1=kmeans_model_birch1.fit_predict(birch1_data)

#Birch2
kmeans_model_birch2=KMeans(n_clusters=50,random_state=561)
kmeans_clusters_birch2=kmeans_model_birch2.fit_predict(birch2_data)

#Birch3
kmeans_model_birch3=KMeans(n_clusters=50,random_state=561)
kmeans_clusters_birch3=kmeans_model_birch3.fit_predict(birch1_data)



# Visualizing K-means Clustering of Birch sets
plt.figure(figsize=(20,10))

#Birch1
plt.subplot(1,3,1)
plt.scatter(birch1_data["X1"],birch1_data["X2"],c=kmeans_clusters_birch1,cmap="viridis",marker=".")
plt.title("Kmeans-Birch 1",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)


#Birch2
plt.subplot(1,3,2)
plt.scatter(birch2_data["X1"],birch2_data["X2"],c=kmeans_clusters_birch2,cmap="viridis",marker=".")
plt.title("Kmeans-Birch 2",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

#Birch3
plt.subplot(1,3,3)
plt.scatter(birch3_data["X1"],birch3_data["X2"],c=kmeans_clusters_birch3,cmap="viridis",marker=".")
plt.title("Kmeans--Birch 3",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)


plt.tight_layout()
plt.show()



# Applying Gaussian Mixture Model clustering on Birch sets

#Birch 1
gmm_model_birch1=GaussianMixture(n_components=50,random_state=561)
gmm_clusters_birch1=gmm_model_birch1.fit_predict(birch1_data)

#Birch 2
gmm_model_birch2=GaussianMixture(n_components=50,random_state=561)
gmm_clusters_birch2=gmm_model_birch2.fit_predict(birch2_data)

#Birch 3
gmm_model_birch3=GaussianMixture(n_components=50,random_state=561)
gmm_clusters_birch3=gmm_model_birch3.fit_predict(birch3_data)

gmm_clusters_birch1

# Visualizing GMM Clustering of Birch sets
plt.figure(figsize=(20,10))

#Birch1
plt.subplot(1,3,1)
plt.scatter(birch1_data["X1"],birch1_data["X2"], c=gmm_clusters_birch1, cmap="viridis",marker=".")
plt.title("GMM-Birch 1",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

#Birch2
plt.subplot(1,3,2)
plt.scatter(birch2_data["X1"],birch2_data["X2"],c=gmm_clusters_birch2,cmap="viridis",marker=".")
plt.title("GMM-Birch 2",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

#Birch3
plt.subplot(1,3,3)
plt.scatter(birch3_data["X1"],birch3_data["X2"],c=gmm_clusters_birch3, cmap="viridis",marker=".")
plt.title("GMM-Birch 3",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)


plt.tight_layout()
plt.show()





b1_ground_trouth_path="/content/drive/Birch Sets/b1-gt.txt"
b2_ground_trouth_path="/content/drive/b2-gt.txt"
#b3_ground_trouth_path="/content/drive/b3-gt.txt.txt"


b1_ground_truth_labels=pd.read_csv(b1_ground_trouth_path,sep="\+",header=None)
b2_ground_truth_labels=pd.read_csv(b2_ground_trouth_path,sep="\+",header=None)
#b3_ground_truth_labels=pd.read_csv(b3_ground_trouth_path,sep="\+",header=None)

print(b1_ground_truth_labels.shape)
print(b1_ground_truth_labels.head())

# Extract the labels of Birch Sets

#Birch1
b1_ground_truth_labels_final=b1_ground_truth_labels.iloc[4:].reset_index(drop=True)
b2_ground_truth_labels_final=b2_ground_truth_labels.iloc[4:].reset_index(drop=True)

print(b1_ground_truth_labels_final.shape)
print(b1_ground_truth_labels_final.head())

print(b2_ground_truth_labels_final.shape)
print(b2_ground_truth_labels_final.head())

# Fix the labels format of Birch Sets

#Birch 1
b1_ground_truth_labels_final=b1_ground_truth_labels_final[0].astype(int)

#Birch 2
b2_ground_truth_labels_final=b2_ground_truth_labels_final[0].astype(int)

print(b1_ground_truth_labels_final)

print(b2_ground_truth_labels_final)

# Calcualte ARI and NMI Metrics for Birch Sets

############# Birch1
#Kmeans
ari_kmeans_b1=adjusted_rand_score(b1_ground_truth_labels_final,kmeans_clusters_birch1)
nmi_kmeans_b1=normalized_mutual_info_score(b1_ground_truth_labels_final,kmeans_clusters_birch1)

#GMM
ari_gmm_b1=adjusted_rand_score(b1_ground_truth_labels_final,gmm_clusters_birch1)
nmi_gmm_b1=normalized_mutual_info_score(b1_ground_truth_labels_final,gmm_clusters_birch1)

############# Birch2
#Kmeans
ari_kmeans_b2=adjusted_rand_score(b2_ground_truth_labels_final,kmeans_clusters_birch2)
nmi_kmeans_b2=normalized_mutual_info_score(b2_ground_truth_labels_final,kmeans_clusters_birch2)

#GMM
ari_gmm_b2=adjusted_rand_score(b2_ground_truth_labels_final,gmm_clusters_birch2)
nmi_gmm_b2=normalized_mutual_info_score(b2_ground_truth_labels_final,gmm_clusters_birch2)

#Birch1
print(ari_kmeans_b1,nmi_kmeans_b1)
print(ari_gmm_b1,nmi_gmm_b1)

#Birch2
print(ari_kmeans_b2,nmi_kmeans_b2)
print(ari_gmm_b2,nmi_gmm_b2)

"""**G2 Sets Clustering Analysis**"""

# Paths to the uploaded files
g2_path="/content/drive/g2-1024-100.txt"

g2_data = pd.read_csv(g2_path,sep="\s+", header=None)

print(g2_data.shape)
print(g2_data.head())

#Data reduction with UMAP (Uniform Manifold Approximation and Projection)
umap_model_g2=UMAP(n_neighbors=15,min_dist=0.1,n_components=2)

umap_data_g2=umap_model_g2.fit_transform(g2_data)

umap_data_g2

# Plot reduced data with UMAP
plt.figure(figsize=(20,10))
plt.scatter(umap_data_g2[:,0],umap_data_g2[:,1])
plt.xlabel('UMAP 1',fontsize=label_fontsize)
plt.ylabel("UMPA 2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)
#plt.title("UMAP Dimensionality Reduction")
plt.show()

# Applying K-means clustering on G2 sets
kmeans_model_g2=KMeans(n_clusters=2,random_state=561)
kmeans_clusters_g2=kmeans_model_g2.fit_predict(umap_data_g2)

kmeans_clusters_g2

# Applying GMM clustering on G2 sets
gmm_model_g2=GaussianMixture(n_components=2,random_state=561)
gmm_clusters_g2=gmm_model_g2.fit_predict(umap_data_g2)

gmm_clusters_g2

len(gmm_clusters_g2)



g2_ground_trouth_path="/content/drive/g2-1024-100-round truth partitions.txt"

g2_ground_truth_labels=pd.read_csv(g2_ground_trouth_path,sep="\+",header=None)

print(g2_ground_truth_labels.shape)
print(g2_ground_truth_labels.head())

# Extract the labels of G2 Sets
g2_ground_truth_labels_final=g2_ground_truth_labels.iloc[4:].reset_index(drop=True)

print(g2_ground_truth_labels_final.shape)
print(g2_ground_truth_labels_final.head())

# Fix the labels format of G2 Sets

g2_ground_truth_labels_final=g2_ground_truth_labels_final[0].astype(int)

print(g2_ground_truth_labels_final)

# Calcualte ARI and NMI Metrics for G2 Sets
#Kmeans
ari_kmeans_g2=adjusted_rand_score(g2_ground_truth_labels_final,kmeans_clusters_g2)
nmi_kmeans_g2=normalized_mutual_info_score(g2_ground_truth_labels_final,kmeans_clusters_g2)

#GMM
ari_gmm_g2=adjusted_rand_score(g2_ground_truth_labels_final,gmm_clusters_g2)
nmi_gmm_g2=normalized_mutual_info_score(g2_ground_truth_labels_final,gmm_clusters_g2)

#Birch1
print(ari_kmeans_g2,nmi_kmeans_g2)
print(ari_gmm_g2,nmi_gmm_g2)

# Visualizing Clustering of G2 sets
plt.figure(figsize=(20,10))

#Kmeans
plt.subplot(1,2,1)
plt.scatter(umap_data_g2[:,0],umap_data_g2[:,1], c=kmeans_clusters_g2, cmap="viridis",marker=".")
plt.title("Kmeans Clustering of G2 Sets",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

#GMM
plt.subplot(1,2,2)
plt.scatter(umap_data_g2[:,0],umap_data_g2[:,1], c=gmm_clusters_g2, cmap="viridis",marker=".")
plt.title("GMM Clustering of G2 Sets",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

plt.tight_layout()
plt.show()





"""**Dim Sets Clustering Analysis**"""

# Paths to the uploaded files
d32_path="/content/drive/dim032.txt"
d1024_path="/content/drive/dim1024.txt"

d32_data = pd.read_csv(d32_path,sep="\s+", header=None)
d1024_data = pd.read_csv(d1024_path,sep="\s+", header=None)

print(d32_data.shape)
print(d32_data.head())

print(d1024_data.shape)
print(d1024_data.head())

#Data reduction of DIM Sets with UMAP (Uniform Manifold Approximation and Projection)

# Initialize a UMAP model
umap_model_d32=UMAP(n_neighbors=15,min_dist=0.1,n_components=2)
umap_model_d1024=UMAP(n_neighbors=15,min_dist=0.1,n_components=2)

# Apply the UMAP model on DIM Sets
umap_data_d32=umap_model_d32.fit_transform(d32_data)
umap_data_d1024=umap_model_d1024.fit_transform(d1024_data)

#umap_data_d32=umap_model_g2.fit_transform(d32_data)

umap_data_d32

# Visualizing DIM Sets after reduction
plt.figure(figsize=(20,10))

# Dim032 with UMAP
plt.subplot(1,2,1)
plt.scatter(umap_data_d32[:,0],umap_data_d32[:,1],color="red")
plt.title("UMAP-Dim032",fontsize=title_fontsize)
plt.xlabel('UMAP 1',fontsize=label_fontsize)
plt.ylabel("UMPA 2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)


# Dim1024 with UMAP
plt.subplot(1,2,2)
plt.scatter(umap_data_d1024[:,0],umap_data_d1024[:,1])
#plt.title("UMAP Dimensionality Reduction of DIM Sets-Dim032")
plt.xlabel('UMAP 1',fontsize=label_fontsize)
plt.ylabel("UMPA 2",fontsize=label_fontsize)
plt.title("UMAP-Dim1024",fontsize=title_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

plt.tight_layout()
plt.show()

# Applying K-means clustering on Dim sets with 16 clusters

#Dim 32
kmeans_model_d32=KMeans(n_clusters=16,random_state=561)
kmeans_clusters_d32=kmeans_model_d32.fit_predict(umap_data_d32)

#Dim 1024
kmeans_model_d1024=KMeans(n_clusters=16,random_state=561)
kmeans_clusters_d1024=kmeans_model_d1024.fit_predict(umap_data_d1024)

kmeans_clusters_d32

# Applying GMM clustering on G2 sets with 16 clusters

# Dim32
gmm_model_d32=GaussianMixture(n_components=16,random_state=561)
gmm_clusters_d32=gmm_model_d32.fit_predict(umap_data_d32)

# Dim1024
gmm_model_d1024=GaussianMixture(n_components=16,random_state=561)
gmm_clusters_d1024=gmm_model_d1024.fit_predict(umap_data_d1024)

kmeans_clusters_d32

gmm_clusters_d32

len(gmm_clusters_d32)



# Load DIM Sets ground truth

d32_ground_trouth_path="/content/drive/dim032.p.txt"
d32_ground_truth_labels=pd.read_csv(d32_ground_trouth_path,sep="\+",header=None)

d1024_ground_trouth_path="/content/drive/dim1024.p.txt"
d1024_ground_truth_labels=pd.read_csv(d1024_ground_trouth_path,sep="\+",header=None)

print(d32_ground_truth_labels.shape)
print(d32_ground_truth_labels.head(10))

print(d1024_ground_truth_labels.shape)
print(d1024_ground_truth_labels.head(10))

# Extract the labels of DIM  Sets
d32_ground_truth_labels_final=d32_ground_truth_labels.iloc[5:].reset_index(drop=True)
d1024_ground_truth_labels_final=d1024_ground_truth_labels.iloc[5:].reset_index(drop=True)

print(d32_ground_truth_labels_final.shape)
print(d32_ground_truth_labels_final.head())

# Fix the labels format of DIM Sets
d32_ground_truth_labels_final=d32_ground_truth_labels_final[0].astype(int)
d1024_ground_truth_labels_final=d1024_ground_truth_labels_final[0].astype(int)

print(d32_ground_truth_labels_final)

# Calcualte ARI and NMI Metrics for Dim Sets

######### Dim032
#Kmeans
ari_kmeans_d32=adjusted_rand_score(d32_ground_truth_labels_final,kmeans_clusters_d32)
nmi_kmeans_d32=normalized_mutual_info_score(d32_ground_truth_labels_final,kmeans_clusters_d32)

#GMM
ari_gmm_d32=adjusted_rand_score(d32_ground_truth_labels_final,gmm_clusters_d32)
nmi_gmm_d32=normalized_mutual_info_score(d32_ground_truth_labels_final,gmm_clusters_d32)

######### Dim1024
#Kmeans
ari_kmeans_d1024=adjusted_rand_score(d1024_ground_truth_labels_final,kmeans_clusters_d1024)
nmi_kmeans_d1024=normalized_mutual_info_score(d1024_ground_truth_labels_final,kmeans_clusters_d1024)

#GMM
ari_gmm_d1024=adjusted_rand_score(d1024_ground_truth_labels_final,gmm_clusters_d1024)
nmi_gmm_d1024=normalized_mutual_info_score(d1024_ground_truth_labels_final,gmm_clusters_d1024)

#Dim1024
print(ari_kmeans_d1024,nmi_kmeans_d1024)
print(ari_gmm_d1024,nmi_gmm_d1024)

# Visualizing Kmeans Clustering of Dim sets
plt.figure(figsize=(16,8))

#Dim032
plt.subplot(1,2,1)
plt.scatter(umap_data_d32[:,0],umap_data_d32[:,1], c=kmeans_clusters_d32, cmap="viridis",marker=".")
plt.title("Kmeans-Dim32",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

#Dim1024
plt.subplot(1,2,2)
plt.scatter(umap_data_d1024[:,0],umap_data_d1024[:,1], c=kmeans_clusters_d1024, cmap="viridis",marker=".")
plt.title("Kmeans-Dim1024",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)


plt.tight_layout()
plt.show()

# Visualizing GMM Clustering  of Dim sets
plt.figure(figsize=(20,10))

#Dim032
plt.subplot(1,2,1)
plt.scatter(umap_data_d32[:,0],umap_data_d32[:,1], c=gmm_clusters_d32, cmap="viridis",marker=".")
plt.title("GMM-Dim32",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

# Dim 1024
plt.subplot(1,2,2)
plt.scatter(umap_data_d1024[:,0],umap_data_d1024[:,1], c=gmm_clusters_d1024, cmap="viridis",marker=".")
plt.title("GMM-Dim1024",fontsize=title_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

plt.tight_layout()
plt.show()







"""**Unbalance Sets Clustering Analysis**"""

# Paths to the uploaded files
ub_path="/content/drive/unbalance.txt"

ub_data = pd.read_csv(ub_path,sep="\s+", header=None, names=["X1", "X2"])

print(ub_data.shape)
print(ub_data.head())

# Visualizing Unbalanced
plt.figure(figsize=(20,10))

plt.scatter(ub_data["X1"],ub_data["X2"],color="red")
plt.xlabel('X1',fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
#plt.title("UMAP Dimensionality Reduction of DIM Sets-Dim032")
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)

plt.show()

# Applying  clustering Algorithms  on Unbalanced sets with 8 clusters

#Kmeans
kmeans_model_ub=KMeans(n_clusters=8,random_state=56)
kmeans_clusters_ub=kmeans_model_ub.fit_predict(ub_data)

# GMM
gmm_model_ub=GaussianMixture(n_components=8,random_state=56)
gmm_clusters_ub=gmm_model_ub.fit_predict(ub_data)

print(kmeans_clusters_ub)
print(gmm_clusters_ub)



# Load Unbalance Set ground truth
ub_ground_trouth_path="/content/drive/unbalance-gt.pa.txt"
ub_ground_truth_labels=pd.read_csv(ub_ground_trouth_path,sep="\+",header=None)

print(ub_ground_truth_labels.shape)
print(ub_ground_truth_labels.head(10))

# Extract the labels of Unbalance Set
ub_ground_truth_labels_final=ub_ground_truth_labels.iloc[4:].reset_index(drop=True)

print(ub_ground_truth_labels_final.shape)
print(ub_ground_truth_labels_final.head())

# Fix the labels format of Unbalance Set
ub_ground_truth_labels_final=ub_ground_truth_labels_final[0].astype(int)

print(ub_ground_truth_labels_final)

# Calcualte ARI and NMI Metrics for Unbalance Set

######### Dim032
#Kmeans
ari_kmeans_ub=adjusted_rand_score(ub_ground_truth_labels_final,kmeans_clusters_ub)
nmi_kmeans_ub=normalized_mutual_info_score(ub_ground_truth_labels_final,kmeans_clusters_ub)

#GMM
ari_gmm_ub=adjusted_rand_score(ub_ground_truth_labels_final,gmm_clusters_ub)
nmi_gmm_ub=normalized_mutual_info_score(ub_ground_truth_labels_final,gmm_clusters_ub)

#Dim1024
print(ari_kmeans_ub,nmi_kmeans_ub)
print(ari_gmm_ub,nmi_gmm_ub)

# Visualizing Clustering of Unbalance Set
plt.figure(figsize=(20,10))

#Kmeans
plt.subplot(1,2,1)
plt.scatter(ub_data["X1"],ub_data["X2"], c=kmeans_clusters_ub, cmap="viridis",marker=".")
plt.title("Kmeans- Unbalance Set",fontsize=label_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)


#GMM
plt.subplot(1,2,2)
plt.scatter(ub_data["X1"],ub_data["X2"], c=gmm_clusters_ub, cmap="viridis",marker=".")
plt.title("GMM- Unbalance Set",fontsize=label_fontsize)
plt.xlabel("X1",fontsize=label_fontsize)
plt.ylabel("X2",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=ticks_fontsize,length=tick_length, width=tick_width)


plt.tight_layout()
plt.show()

