import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import pandas

a = pandas.read_csv("nuoret1.csv")
print (a)
print (a["Indikaattori"])
b = pandas.pivot_table(a,index=["Indikaattori", "Sukupuoli", "aidinkieli"])
print(b)
c = (b.loc["Ammattikorkeakoulutuksessa"].loc["Miehet"].loc["Suomi"])
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


# We have x and y lists. The lists are both the same length
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
print("x length: ", len(x), " y leghth: ", len(y))

slope, intercept, r, p, std_err = stats.linregress(x,y)


def func(x):
    return slope * x + intercept

linmodel = list(map(func, x))

plt.scatter(x,y)
plt.plot(x, linmodel)
plt.show()
