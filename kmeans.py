import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import random
#%matplotlib inline

# generate some random data. This data is used later on to search for the K-means
# The array is going to be 0...50 in one part of the graph, 51...100 in the other.
x = -2 * np.random.rand(100,2)
x1 = 1 + 2 * np.random.rand(50,2)
x[50:100, :] = x1

# Let's shuffle the data around. This way we can see later on the clustering better
x=shuffle(x,random_state=0)

# Set the amount of wanted clusters. 
clusters=2
Kmean = KMeans(n_clusters=clusters)
Kmean.fit(x)

# Print the cluster centers
print("cluster centers")
print(Kmean.cluster_centers_[0])
print(Kmean.cluster_centers_[0][0])
print(Kmean.cluster_centers_[0][1])
print(Kmean.cluster_centers_[1])
print(Kmean.cluster_centers_[1][0])
print(Kmean.cluster_centers_[1][1])

# Create a scatter plot with the datapoints as well as the calculated cluster centers.
training_points=plt.scatter(x[ : , 0], x[ :, 1], s = 50, c = 'b')
kmean1_point=plt.scatter(Kmean.cluster_centers_[0][0],Kmean.cluster_centers_[0][1], s=200, c='g', marker='s')
kmean2_point=plt.scatter(Kmean.cluster_centers_[1][0],Kmean.cluster_centers_[1][1], s=200, c='r', marker='s')

# Print the data along with the clusters they belong to (0 or 1)
print(x)
print(Kmean.labels_)


# The clusters have been created and as such we can then see to which cluster a new point belongs to

print("testing with new data")
sample_test = np.array([-3.0,-3.0])
sample_test=sample_test.reshape(1,-1)
# Where does -3,-3 fit?
print (sample_test)
sample_group=Kmean.predict(sample_test)
print(sample_group)
# Add the -3,-3 to scatter
sample_test_point = plt.scatter(sample_test[0][0],sample_test[0][1], s=100, c='m', marker='^')

# where does a random -2...3, -2...3 value fit?
x_random = np.random.rand()*5-2
y_random = np.random.rand()*5-2
random_test = np.array([x_random, y_random])
random_test = random_test.reshape(1,-1)
print(random_test)
random_group=Kmean.predict(random_test)
print(random_group)
# Add the random point to the list
random_test_point = plt.scatter(random_test[0][0],random_test[0][1], s=100, c='y', marker='^')
# Show the scatter plot.
plt.legend([training_points, kmean1_point, kmean2_point, sample_test_point, random_test_point],["training points", "K-mean 1", "K-mean 2", "-3,-3 test point, group: "+str(sample_group), "random test point, group: "+str(random_group)])
plt.show()


