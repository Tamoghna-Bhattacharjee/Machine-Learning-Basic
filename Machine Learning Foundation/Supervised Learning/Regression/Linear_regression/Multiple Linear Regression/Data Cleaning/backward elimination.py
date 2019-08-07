import numpy, pandas
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from matplotlib import pyplot
import seaborn as sns


data = pandas.read_csv('50_Startups.csv')

x = data.iloc[:, :-1]
y = data.iloc[:, -1].values

# encoding dummy variable

label_encoding = LabelEncoder()
x.iloc[:, -1] = label_encoding.fit_transform(x.iloc[:, -1])

one_hot_encoder = OneHotEncoder(categorical_features=[-1])
x = one_hot_encoder.fit_transform(x).toarray()


# replace the 1st dummy variable by intercept column which is must for OLS class
x[:, 0] = numpy.ones(x.shape[0])


# Backward elimination
x_optimal = x.copy()


while True:
    ols = sm.OLS(endog=y, exog=x_optimal).fit()
    p = list(ols.pvalues)

    max_p = max(p)
    if max_p <= 0.05:
        break

    remove_col_ind = p.index(max_p)
    x_optimal = numpy.delete(x_optimal, remove_col_ind, axis=1)


x_train, x_test, y_train, y_test = train_test_split(x_optimal, y, random_state=101, test_size=0.2)


lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

y_pred = lin_reg.predict(x_test)


r_square = metrics.explained_variance_score(y_test, y_pred)
print(r_square)
# pyplot.scatter(y_test, y_pred)
# sns.scatterplot(y_test, y_pred)
# pyplot.show()

