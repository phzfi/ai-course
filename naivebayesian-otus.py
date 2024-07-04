
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



# ikä (v), pituus (cm), paino (kg), sukupuoli (0 = mies, 1 = nainen)
kaikki = [ 
    [1, 50, 10, 0],
    [2, 60, 20, 0],
    [3, 70, 30, 0],
    [1, 70, 20, 1],
    [2, 80, 25, 1],
    [3, 90, 30, 1]
]

# kaikki -> data ja luokittelu

#data = X
data = [ 
    [1, 50, 10],
    [2, 60, 20],
    [3, 70, 30],
    [1, 70, 20],
    [2, 80, 25],
    [3, 90, 30]
]



# luokka = y

luokka = [0,0,0,1,1,1]

#data[0] luokka on luokka[0]


#plt.scatter(X[:, 0], X[:, 1], c=y, marker="*")
#plt.show()

X = data
y = luokka

print(X)
print(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

print("koulutusdata:")
print(X_train)
print(y_train)
print("testidata:")
print(X_test)
print(y_test)



# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X, y)

otus = [4,100,35]

# Predict Output
#predicted = model.predict([X_test[0]])
predicted = model.predict([otus])




#print("Actual Value:", y_test[0])
print("Predicted Value:", predicted[0])