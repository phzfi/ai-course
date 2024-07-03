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
print (c)


slope, intercept, r, p, std_err = stats.linregress(years,c)


def func(x):
    return slope * x + intercept

linmodel = list(map(func, years))

plt.scatter(years,c)
plt.plot(years, linmodel)
plt.show()
