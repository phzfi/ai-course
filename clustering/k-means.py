import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# Here you would read in the data you want to use
# Such as the following. The data is bad for this, but you can still try to use it.

df = pandas.read_csv("clustering/student_mental_health_burnout_1M.csv")
Kmean = KMeans(n_clusters=3)
Kmean.fit(df)

