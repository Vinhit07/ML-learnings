from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
print(iris.data[10])
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN classifier
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Predict on the test set
predictions = knn_classifier.predict(X_test)

# Print correct and wrong predictions
correct_predictions = 0
wrong_predictions = 0

for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        print(f"Correct prediction: Predicted class - {predictions[i]}, Actual class - {y_test[i]}")
        correct_predictions += 1
    else:
        print(f"Wrong prediction: Predicted class - {predictions[i]}, Actual class - {y_test[i]}")
        wrong_predictions += 1

# Calculate and print accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)
print("Number of correct predictions:", correct_predictions)
print("Number of wrong predictions:", wrong_predictions)
