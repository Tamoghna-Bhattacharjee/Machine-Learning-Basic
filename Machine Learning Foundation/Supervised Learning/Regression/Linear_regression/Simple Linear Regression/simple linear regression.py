import numpy, pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib import pyplot
import seaborn as sns


dataset = pandas.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

pyplot.scatter(x_test, y_test)
pyplot.plot(x_test, y_pred)
pyplot.show()