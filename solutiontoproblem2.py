import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from sklearn.mixture import GaussianMixture
import sklearn.model_selection as model_selection

# Read the data
df = pd.read_csv("./result.csv")

# Separate the grade column
grade = df["grade"].tolist()
grade_unique = list(set(grade))
grade_colors = ["r", "g", "b"]

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

# calculating our new axis
pc1 = X.dot(eig_vectors[:, 0])
pc2 = X.dot(eig_vectors[:, 1])

X = np.vstack((pc1, pc2)).T
X = np.array(X)

transformed_grade = []
for i in range(len(grade)):
    if grade[i] == 'A':
        transformed_grade.append(+2)
    elif grade[i] == 'B':
        transformed_grade.append(+1)
    else:
        transformed_grade.append(0)
grade = copy.deepcopy(transformed_grade)
grade = np.array(grade)
n_classes = 3

# TODO 1: split the data into training and test sets (Hint: Use model_selection in sklearn.model_selection)
# TODO 1: The output after splitting should have names: X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, grade, test_size=None, train_size=None,
                                                                    random_state=None, shuffle=True, stratify=None)

NA = 0
NB = 0
NC = 0
sumA1 = 0
sumA2 = 0
sumB1 = 0
sumB2 = 0
sumC1 = 0
sumC2 = 0

for i in range(len(X_train)):
    if y_train[i] == 2:
        NA += 1
        sumA1 += X_train[i][0]
        sumA2 += X_train[i][1]
    elif y_train[i] == 1:
        NB += 1
        sumB1 += X_train[i][0]
        sumB2 += X_train[i][1]
    else:
        NC += 1
        sumC1 += X_train[i][0]
        sumC2 += X_train[i][1]


meanmatrix = [[sumC1 / NC, sumC2/ NC], [sumB1 / NB, sumB2 / NB],[sumA1 / NA, sumA2 / NA]]

# TODO 2: Train your GMM (Hint: Use GaussianMixture in sklearn.mixture. You need to create a class object of GMM, initialize the means, and estimate other parameters)
# TODO 2: The GMM object should have name: estimator
estimator = GaussianMixture(n_components=n_classes, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100,
                            n_init=1, init_params='k-means++', means_init=meanmatrix, random_state=None,
                            warm_start=False, verbose=0, verbose_interval=10).fit(X_train)

# Plot
fig, ax = plt.subplots()
for n, color in enumerate(grade_colors):
    # Plot all data points where the test data are marked with crosses
    data = X[grade == n]
    data_test = X_test[y_test == n]
    plt.scatter(data[:, 0], data[:, 1], s=10, color=color, label=grade_colors[n])
    plt.scatter(data_test[:, 0], data_test[:, 1], marker="x", color=color)

    # Plot the ellipses
    cov = estimator.covariances_[n][:2, :2]
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.rad2deg(np.arctan2(u[1], u[0]))
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ellipse = mpl.patches.Ellipse(estimator.means_[n, :2], v[0], v[1], 180 + angle, color=color)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)

print("Means: ", estimator.means_)
print("Cov: ", estimator.covariances_)
print("Weights: ", estimator.weights_)

# Calculate accuracies and print on the plot
y_train_pred = estimator.predict(X_train)
y_test_pred = estimator.predict(X_test)
train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
plt.text(0.05, 0.9, "Train accuracy: %.2f %%" % train_accuracy, transform=ax.transAxes)
plt.text(0.05, 0.8, "Test accuracy: %.2f %%" % test_accuracy, transform=ax.transAxes)

ax.set_aspect("equal", "datalim")
plt.legend(loc=4)
plt.tight_layout()
plt.show()

# Answer to problem 2-(b):
# Means:
# K1:[1.68956631 -0.83705862], K2:[-0.15974893  0.52926648], K3:[-1.65198753 -0.0236597 ]
# Cov:
# K1:[[ 2.63223275  0.23965969],[ 0.23965969  0.67436553]]
# K2:[[ 0.34560662 -0.01271518],[-0.01271518  0.49441819]]
# K3:[[ 0.1303388  -0.0699402 ],[-0.0699402   0.44494523]]
# Weights:
# K1:0.2802275 K2:0.31780281 K3:0.40196969
