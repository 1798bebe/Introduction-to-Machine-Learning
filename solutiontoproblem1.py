import numpy as np
import pandas as pd
import math

# Read the data
df = pd.read_csv("./result.csv")

# Separate the grade column
grade = df["grade"].tolist()
X = df.drop(columns="grade")

# Standardize the data
X = (X - X.mean()) / X.std(ddof=0)

# Calculating the covariance matrix of the data
covariance = np.cov(X.T)

# Get the Eigenvalues (sorted) and their corresponding eigenvectors using Eigendecomposition
eig_values, eig_vectors = np.linalg.eig(covariance)

idx = eig_values.argsort()[::-1]
eig_values = eig_values[idx]
eig_vectors = eig_vectors[:, idx]

print("\n***  Reconstruction error (RMSE) when using the first two PCs  ***")
pc1 = X.dot(eig_vectors[:, 0])
pc2 = X.dot(eig_vectors[:, 1])

X_projected = np.vstack((pc1, pc2)).T  # append pc2 to pc1 by placing it on the next row
B = np.column_stack([eig_vectors[:, 0], eig_vectors[:, 1]])
X_hat = np.matmul(X_projected,B.T)

rmse_val = (((X_hat-X)**2).mean(axis=0))**(0.5)
print(rmse_val)

print("\n***  Reconstruction error (RMSE) when using the last two PCs  ***")

pc7 = X.dot(eig_vectors[:, 6])
pc8 = X.dot(eig_vectors[:, 7])

X_projected2 = np.vstack((pc7, pc8)).T
B2 = np.column_stack([eig_vectors[:, 6], eig_vectors[:, 7]])
X_hat2 = np.matmul(X_projected2,B2.T)

rmse_val2 = (((X_hat2-X)**2).mean(axis=0))**(0.5)
print(rmse_val2)


## Answer to problem 1-(b):

## The RMSE gets minimized as we maximize the variance of the projection that we retain in the lower dimensional subspace.
## The variance of the data when projected onto the lower dimensional subspace equals the sum of the eigenvalues that are
## associated with the corresponding eigenvectors of the data covariance matrix.
## As in the former case, we chose the two principal components which correspond to the two largest eigenvalues of the data covariance matrix,
## so that the error had been minimized.
## As in the latter case, we chose the two principal components which correspond to the two smallest eigenvalues,
## so that the error had been maximized.



