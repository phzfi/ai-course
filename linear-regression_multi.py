import pandas
from sklearn import linear_model

df = pandas.read_csv("car_data.csv")
print(df)

X = df[['Weight', 'Volume']]
print(X)
y = df['CO2']
print(y)

regr = linear_model.LinearRegression()
regr.fit(X.values, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(regr.coef_)
print(regr.score(X.values, y))

print(predictedCO2) 