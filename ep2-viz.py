# import numpy
import numpy as np

# import DecisionTreeClassifier
from sklearn import tree

# import dataset
from sklearn.datasets import load_iris
iris = load_iris()

# print feature names
# print(iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# print species names
# print(iris.target_names)
# ['setosa' 'versicolor' 'virginica']

# print features of first item in dataset
# print(iris.data[0])

# print species of first item in dataset
# print(iris.target[0])

# iterate over all items in dataset
# for i in range(len(iris.target)):
#     print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

# train a classifier
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# fit using DTC
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# print our target value
print(test_target)

# predict label for new flower
print(clf.predict(test_data))
# should be same as the target value above

# visualize the tree
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)

# create decision tree graph and write out a PDF
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
