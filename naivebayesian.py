
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import make_classification

X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

print(X)
print(y)

plt.scatter(X[:, 0], X[:, 1], c=y, marker="*")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)


# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train)

# Predict Output
predicted = model.predict([X_test[5]])

print("Testidata")
print(X_test)
print(y_test)


print("Actual Value:", y_test[5])
print("Predicted Value:", predicted[0])