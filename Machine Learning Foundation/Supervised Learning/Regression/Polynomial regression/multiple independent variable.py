import numpy, pandas
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn import metrics
from matplotlib import pyplot
import seaborn as sns

df = pandas.read_csv('USA_Housing.csv')

x = df.iloc[:, :-2]
y = df.iloc[:, -2]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

pol_reg = PolynomialFeatures(degree=4)
x_train_poly, x_test_poly = pol_reg.fit_transform(x_train), pol_reg.fit_transform(x_test)

lin_reg = LinearRegression()
lin_reg.fit(x_train_poly, y_train)

y_pred = lin_reg.predict(x_test_poly)

print(metrics.explained_variance_score(y_test, y_pred))
pyplot.scatter(y_test, y_pred)
pyplot.show()

