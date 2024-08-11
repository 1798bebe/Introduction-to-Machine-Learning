import numpy as np
import pandas as pd
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score

# Read the data
df = pd.read_csv("./result.csv")

# Seperate the grade column
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
eig_vectors = eig_vectors[:,idx]

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

#TODO 1: split the data into training and test sets (Hint: Use model_selection in sklearn.model_selection)
#TODO 1: The output after splitting should have names: X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, transformed_grade, test_size=None, train_size=None,
                                                                    random_state=None, shuffle=True, stratify=None)

#TODO 2: Create SVM object with a kernel of your choice (Hint: Use svm in sklearn)
#TODO 2: The SVM object should have name: estimator
estimator=svm.SVC(C=1.0,kernel='linear',coef0=0.0,shrinking=True)
estimator.fit(X_test,y_test)

#TODO 3: Predict the class labels of the text set
#TODO 3: The predicted labels should have name: pred
pred = estimator.predict(X_test)

# Calculate and print the accuracy of the SVM / Show the predictions and ground truth values
accuracy = accuracy_score(y_test, pred)
print('Accuracy:\t', "%.2f %%" % (accuracy*100))
print("Test:\t\t", y_test)
print("Prediction:\t", list(pred))

# Answer to problem 3-(b):
# We used the linear kernel because the linear kernel in general provides much faster performance than other kernels.
# Also, we repetitively examined the performances of many different kernels such as the linear kernel,
# the RBF(Gaussian) kernel,the polynomial kernel of degree 3... etc and we found that for this data set,
# the accuracy provided by the linear kernel was not noticeably bad compared to those of any other kernels.
# Therefore, we concluded that the linear kernel was the best fit for this problem. 