import keras
import theano
import tensorflow
import numpy, pandas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pandas.read_csv('Churn_Modelling.csv')
x = df.iloc[:, 3: 13].values
y = df.iloc[:, 13].values

label_encoder1 = LabelEncoder()
x[:, 1] = label_encoder1.fit_transform(x[:, 1])


