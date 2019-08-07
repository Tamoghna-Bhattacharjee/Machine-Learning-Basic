import numpy, pandas
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot


df = pandas.read_csv('Position_Salaries.csv')

x = df.iloc[:, 1: 2].values
y = df.iloc[:, 2].values

"""Since the data is small train/test split is not done"""

random_forest = RandomForestRegressor(n_estimators=500, max_depth=3, random_state=101)
random_forest.fit(x, y)

print(random_forest.predict([[6.5]]))



x_grid = numpy.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)

pyplot.scatter(x, y, color="red")
pyplot.plot(x_grid, random_forest.predict(x_grid))

pyplot.show()