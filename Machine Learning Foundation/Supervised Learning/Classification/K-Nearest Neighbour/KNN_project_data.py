import pandas
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pandas.read_csv('KNN_Project_Data')
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

sc_x = StandardScaler()
x = sc_x.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

error = []

for i in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=i, p=2, weights='distance')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    error.append(conf_matrix[0, 1] + conf_matrix[1, 0])


plt.figure(figsize=(10, 6))
plt.plot(range(1, 41), error, marker='o', ls='--', mfc='g', ms=10)
plt.ylabel('Error')
plt.xlabel('n_neighbours')
plt.show()

