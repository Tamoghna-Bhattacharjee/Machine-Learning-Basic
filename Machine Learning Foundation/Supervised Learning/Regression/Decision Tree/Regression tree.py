import numpy, pandas
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from matplotlib import pyplot


df = pandas.read_csv('Position_Salaries.csv')

x = df.iloc[:, 1: 2].values
y = df.iloc[:, 2].values

"""Since the data is small train/test split is not done"""

descition_tree = DecisionTreeRegressor(random_state=101, max_depth=5)
descition_tree.fit(x, y)

print(descition_tree.predict([[6.5]]))


x_grid = numpy.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)

pyplot.scatter(x, y, color="red")
pyplot.plot(x_grid, descition_tree.predict(x_grid))

pyplot.show()