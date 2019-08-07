from keras.layers import Dense
from keras.models import Sequential
import theano
import tensorflow as tf
import numpy, pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix


df = pandas.read_csv('Churn_Modelling.csv')
x = df.iloc[:, 3: 13].values
y = df.iloc[:, 13].values

label_encoder = LabelEncoder()
x[:, 1] = label_encoder.fit_transform(x[:, 1])
x[:, 2] = label_encoder.fit_transform(x[:, 2])

one_hot_encoder = OneHotEncoder(categorical_features=[1])
x = one_hot_encoder.fit_transform(x).toarray()
x = x[:, 1:]

standard_scaler = StandardScaler()
x = standard_scaler.fit_transform(x)

x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2, random_state=42)


# ANN
classifier = Sequential()


# adding input layer and first hidden layer
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', input_dim=11, activation='relu'))

# adding 2nd layer
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu'))

# adding output layer
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)


y_pred = classifier.predict(x_test)
y_pred = [0 if i < 0.5 else 1 for i in y_pred]

cm = confusion_matrix(y_test, y_pred)

print(cm)
