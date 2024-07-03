import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import pandas

a = pandas.read_csv("dataset.csv", sep=";")
d=a.drop(columns=['2022'])
print (d)
b = pandas.pivot_table(d,index=["Alue", "Sukupuoli"])

print(b)
c = (b.loc["Eteläinen suurpiiri"].loc["Miehet"])
print (c)
#years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
years = np.arange(1999,2022,1)
print (c)


slope, intercept, r, p, std_err = stats.linregress(years,c)
print(r)

def func(x):
    return slope * x + intercept

linmodel = list(map(func, years))

plt.scatter(years,c)
plt.plot(years, linmodel)
plt.show()
