from scipy.spatial import distance

# defining our custom classifier
class ScrappyKNN():
    # here, we memorize the dataset into our class
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # here, we predict the targets
    def predict(self, X_test):
        predictions = []

        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    # here, we calculate the nearest neighbors using euclidean distance between two points
    def closest(self, row):
        best_dist = distance.euclidean(row, self.X_train[0])
        best_index = 0

        for i in range(1, len(self.X_train)):
            dist = distance.euclidean(row, self.X_train[i])

            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y_train[best_index]

# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

# put the data and targets as X and y to simulate a function
X = iris.data
y = iris.target

# partition the data as train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# get the Scrappy K-Nearest Neighbors Classifier
sknn_classifier = ScrappyKNN()

# fit a curve using SKNN
sknn_classifier.fit(X_train, y_train)

# predict the target on given test data using SKNN
scrappy_predictions = sknn_classifier.predict(X_test)

# display accuracy score or confidence factor
from sklearn.metrics import accuracy_score
print("Accuracy Score using our Scrappy KNN: %s" % accuracy_score(y_test, scrappy_predictions))
