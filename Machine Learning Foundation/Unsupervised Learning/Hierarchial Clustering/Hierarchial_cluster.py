import numpy, pandas
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

df = pandas.read_csv('Mall_Customers.csv')
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
plt.show()

x = df.iloc[:, 3:5].values

# making dendrogram
dendrogram = sch.dendrogram(sch.linkage(x, method='ward', metric='euclidean'))
plt.ylabel('Distance')
plt.show()

hac = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred = hac.fit_predict(x)

# visualization
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], color='r')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], color='g')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], color='y')
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], color='orange')
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], color='b')
plt.show()

