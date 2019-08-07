import numpy, pandas
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn import metrics
from matplotlib import pyplot
import seaborn as sns


df = pandas.read_csv('Position_Salaries.csv')

x = df.iloc[:, 1: 2].values
y = df.iloc[:, 2].values

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# polynomial regression
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))

pyplot.scatter(x, y)
pyplot.plot(x, lin_reg_2.predict(x_poly), color='red', label='Polynomial regression')
pyplot.plot(x, lin_reg.predict(x), color='g', label='Linear regression')
pyplot.legend()
pyplot.show()
