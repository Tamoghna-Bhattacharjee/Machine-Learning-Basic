import numpy, pandas
import seaborn as sns
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

df = pandas.read_csv('titanic_data.csv')
df.drop(labels=['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)


mean_age_1 = numpy.mean(df[df['Pclass'] == 1]['Age'].dropna())
mean_age_2 = numpy.mean(df[df['Pclass'] == 2]['Age'].dropna())
mean_age_3 = numpy.mean(df[df['Pclass'] == 3]['Age'].dropna())
mean_ages = [mean_age_1, mean_age_2, mean_age_3]


def impute_age(cols):
    global mean_ages
    age = cols[0]
    pclass = cols[1]

    if pandas.isnull(age):
        return mean_ages[int(pclass) - 1]
    else:
        return age


df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)
df.dropna(axis=0, inplace=True)

# sns.heatmap(data=df.isnull())
# pyplot.show()

x = df.iloc[:, 1:]
y = df.iloc[:, 0].values

label_encode = LabelEncoder()
x['Sex'] = label_encode.fit_transform(x['Sex'])
x['Embarked'] = label_encode.fit_transform(x['Embarked'])

one_hot_encoder = OneHotEncoder(categorical_features=[1])
x = one_hot_encoder.fit_transform(x).toarray()
x = x[:, 1:]
one_hot_encoder = OneHotEncoder(categorical_features=[1])
x = one_hot_encoder.fit_transform(x).toarray()
x = x[:, 1:]
one_hot_encoder = OneHotEncoder(categorical_features=[7])
x = one_hot_encoder.fit_transform(x).toarray()
x[:, 0] = numpy.ones(x.shape[0])


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=101, test_size=0.25)

logistic_reg = LogisticRegression(random_state=101)
logistic_reg.fit(x_train, y_train)

y_pred = logistic_reg.predict(x_test)

conf_matrix = confusion_matrix(y_test, y_pred)

print(conf_matrix)
