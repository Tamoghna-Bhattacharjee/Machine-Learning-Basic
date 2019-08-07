import numpy, pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib import pyplot
import seaborn as sns

"""
Index(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'],
      dtype='object')

"""
df = pandas.read_csv('USA_Housing.csv')

# ================== splitting of data ====================

x = df.iloc[:, :-2]
y = df.iloc[:, -2]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# ================== splitting of data ====================


# ================== making model ====================

linreg = LinearRegression()
linreg.fit(x_train, y_train)

# ================== making model ====================


# ================== getting coef and intercept ====================

reg_coef_df = pandas.DataFrame(linreg.coef_, x.columns, columns=['coef'])
print(reg_coef_df)
print(linreg.intercept_)

# ================== getting coef and intercept ====================


# ================== prediction ====================

prediction = linreg.predict(x_test)
print(prediction)

# sns.scatterplot(y_test, prediction)
# sns.distplot(y_test - prediction, bins=30)
# pyplot.show()

# ================== prediction ====================


# ================== checking error ====================

mae = metrics.mean_absolute_error(y_test, prediction)
mse = metrics.mean_squared_error(y_test, prediction)
rms = numpy.sqrt(mae)
r_square = metrics.explained_variance_score(y_test, prediction)

print(f"MAE = {mae}\t MSE = {mse}\nRMS = {rms}\t R^2 = {r_square}")

# ================== checking error ====================

