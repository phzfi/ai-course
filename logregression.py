import numpy as np

import matplotlib.pyplot as plt
import pandas

a = pandas.read_csv("nuoret1.csv")
print (a)
print (a["Indikaattori"])
b = pandas.pivot_table(a,index=["Indikaattori", "Sukupuoli", "aidinkieli"])
print(b)
c = (b.loc["Ammattikorkeakoulutuksessa"].loc["Miehet"].loc["Suomi"])
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
print (c)

#Which
x = np.arange(1,14,1)
#x = years
y=c
plt.scatter(x, y)
#fit the model
fit = np.polyfit(np.log(x), y, 1)

#view the output of the model
print(fit)

def logf(x):
    return fit[1]+(fit[0] * np.log(x))  

print (logf(12))

logmodel = list(map(logf, x))
print (logmodel)

known_x = 2033
prediction = logf(known_x)
print("Predicted for " + str(known_x) + " = " + str(prediction))
print (prediction)


plt.scatter(x,y)

plt.plot(x, logmodel)
plt.show()