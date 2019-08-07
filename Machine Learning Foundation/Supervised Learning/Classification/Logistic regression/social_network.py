import numpy, pandas
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from statsmodels.regression.linear_model import OLS

df = pandas.read_csv('Social_Network_Ads.csv')

x = df.iloc[:, 1:4].values
y = df.iloc[:, 4].values

# data encoding
label_encode = LabelEncoder()
x[:, 0] = label_encode.fit_transform(x[:, 0])
one_hot_encode = OneHotEncoder(categorical_features=[0])
x = one_hot_encode.fit_transform(x).toarray()
x[:, 0] = numpy.ones(x.shape[0])


# standardising the cols
standard_scaler = StandardScaler()
x[:, 2:] = standard_scaler.fit_transform(x[:, 2:])


# to know which cols to remove
ols = OLS(endog=y, exog=x).fit()
print(ols.summary())


# dropping col that are having high p-values and the constant col
x = numpy.delete(x, [0, 1], axis=1)


# data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=101, test_size=0.25)

# building classifying model
logistic_regressor = LogisticRegression(random_state=101)
logistic_regressor.fit(x_train, y_train)


# getting the prediction and design confusion matrix
y_pred = logistic_regressor.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# visualisation
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = numpy.meshgrid(
            numpy.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1, 0.01),
            numpy.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1, 0.01)
        )
plt.contourf(x1, x2, logistic_regressor.predict(numpy.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))


plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

plt.scatter(x_set[y_set == 0, 0], x_set[y_set == 0, 1], color='r', alpha=1, label=0)
plt.scatter(x_set[y_set == 1, 0], x_set[y_set == 1, 1], color='g', alpha=1, label=1)

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

