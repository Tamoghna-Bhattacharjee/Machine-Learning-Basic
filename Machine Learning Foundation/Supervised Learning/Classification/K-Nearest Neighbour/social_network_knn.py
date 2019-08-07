import numpy, pandas
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from statsmodels.regression.linear_model import OLS
from sklearn.neighbors import KNeighborsClassifier


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
standard_scalar = StandardScaler()
x[:, 2:] = standard_scalar.fit_transform(x[:, 2:])


# to know which cols to remove
ols = OLS(endog=y, exog=x).fit()
print(ols.summary())


# dropping col that are having high p-values and the constant col
x = numpy.delete(x, [0, 1], axis=1)


# data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)


# K-NN
classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2, metric='minkowski')
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# visualisation on train set
plt.subplot(1, 2, 1)
x_set, y_set = x_train, y_train
x1, x2 = numpy.meshgrid(
            numpy.arange(x_set[:, 0].min() - 1, x_set[:, 0].max()+1, 0.01),
            numpy.arange(x_set[:, 1].min() - 1, x_set[:, 1].max()+1, 0.01)
        )

grid_co_ordinate_metrix = numpy.array([x1.ravel(), x2.ravel()]).T
plt.contourf(x1, x2, classifier.predict(grid_co_ordinate_metrix).reshape(x1.shape),
             cmap='RdYlGn', alpha=0.75)

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

plt.scatter(x_set[y_set == 0, 0], x_set[y_set == 0, 1], color='r', label='0')
plt.scatter(x_set[y_set == 1, 0], x_set[y_set == 1, 1], color='g', label='1')

plt.title('KNN (Train set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
# plt.show()


# visualisation on test set
x_set, y_set = x_test, y_test
plt.subplot(1, 2, 2)
x1, x2 = numpy.meshgrid(
            numpy.arange(x_set[:, 0].min() - 1, x_set[:, 0].max()+1, 0.01),
            numpy.arange(x_set[:, 1].min() - 1, x_set[:, 1].max()+1, 0.01)
        )

grid_co_ordinate_metrix = numpy.array([x1.ravel(), x2.ravel()]).T
plt.contourf(x1, x2, classifier.predict(grid_co_ordinate_metrix).reshape(x1.shape),
             cmap='RdYlGn', alpha=0.75)

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

plt.scatter(x_set[y_set == 0, 0], x_set[y_set == 0, 1], color='r', label='0')
plt.scatter(x_set[y_set == 1, 0], x_set[y_set == 1, 1], color='g', label='1')

plt.title('KNN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

