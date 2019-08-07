from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

import pandas

data = pandas.read_csv('Data.csv')
x = data.iloc[:, :-1]
y = data.iloc[:, 3]

# =========== handling missing data ================
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x.iloc[:, 1:])
x.iloc[:, 1:] = imputer.transform(x.iloc[:, 1:])
# =========== handling missing data ================



# =========== LabelEncoder ================
labelencoder_x = LabelEncoder()
x.iloc[:, 0] = labelencoder_x.fit_transform(x.iloc[:, 0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
# =========== LabelEncoder ================




# =========== OneHotEncoder ================
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()
x = pandas.DataFrame(x)
# =========== OneHotEncoder ================


# =========== scaling data ================
minmaxscaler = MinMaxScaler(feature_range=(0,1))
x.iloc[:, 3:] = minmaxscaler.fit_transform(x.iloc[:, 3:])

# =========== scaling data ================


# =========== creating training and test set ================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# print(x_train)
# =========== creating training and test set ================


print(x)
