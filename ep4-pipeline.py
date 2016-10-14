# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

# put the data and targets as X and y to simulate a function
X = iris.data
y = iris.target

# partition the data as train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# get the Decision Tree Classifier
from sklearn import tree
dtc = tree.DecisionTreeClassifier()

# fit a curve using DTC
dtc.fit(X_train, y_train)

# predict the target on given test data using DTC
dtc_predictions = dtc.predict(X_test)

# get the K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# fit a curve using KNN
knn.fit(X_train, y_train)

# predict the target on given test data using KNN
knn_predictions = knn.predict(X_test)

# display accuracy score or confidence factor
from sklearn.metrics import accuracy_score
print("Accuracy Score using Decision Tree Classifier: %s" % accuracy_score(y_test, dtc_predictions))
print("Accuracy Score using K-Nearest Neighbors Classifier: %s" % accuracy_score(y_test, knn_predictions))
