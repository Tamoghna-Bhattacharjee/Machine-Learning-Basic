import numpy, pandas
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

df = pandas.read_csv('Mall_Customers.csv')
x = df.iloc[:, 3: 5].values

# wssr = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, init='k-means++', random_state=42)
#     kmeans.fit(x)
#     wssr.append(kmeans.inertia_)
#
#
# plt.plot(range(1, 11), wssr, marker='o')
# plt.show()
# from graph above the optimal k = 5

kmeans = KMeans(n_clusters=5, max_iter=300, n_init=10, init='k-means++')
kmeans.fit(x)
y_pred = kmeans.predict(x)


plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], color='r')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], color='g')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], color='y')
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], color='orange')
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], color='b')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=400, color='pink')
plt.show()

