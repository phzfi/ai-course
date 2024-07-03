import numpy as np
import matplotlib.pyplot as plt
import pandas

# Here you would read in the data you want to use
# Such as the following. The data is bad for this, but you can still try to use it.

a = pandas.read_csv("nuoret1.csv")
print (a)
print (a["Indikaattori"])
b = pandas.pivot_table(a,index=["Indikaattori", "Sukupuoli", "aidinkieli"])
print(b)
c = (b.loc["Ammattikorkeakoulutuksessa"].loc["Miehet"].loc["Suomi"])
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
print (c)



#x = np.arange(2010, 2023, 1)
#y = c

x = np.arange(2010, 2030, 1)
y = np.array([1, 3, 5, 7, 9, 12, 15, 19, 23, 28,
              33, 38, 44, 50, 56, 64, 73, 84, 97, 113])


plt.scatter(x, y)


#fit the model
fit = np.polyfit(x, np.log(y), 1)

#view the output of the model
print(fit)

def expf(x):
    return np.exp(fit[1])*np.power(np.exp(fit[0]),x)


expmodel = list(map(expf, x))

plt.scatter(x,y)
plt.plot(x,expmodel)

plt.show()