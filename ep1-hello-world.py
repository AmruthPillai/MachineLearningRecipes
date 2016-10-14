from sklearn import tree

# smooth is 1, bumpy is 0
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# apples are 0, oranges are 1
labels = [0, 0, 1, 1]

# using DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()

# fitting the line using test data
clf = clf.fit(features, labels)

# predict if it's an apple or an orange
# it should print out [1] for orange, since weight is greater
print(clf.predict([160, 0]))
