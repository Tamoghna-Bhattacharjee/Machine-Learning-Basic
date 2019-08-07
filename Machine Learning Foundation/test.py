from matplotlib import pyplot as plt

x = [i for i in range(6)]
y1 = [-0.5, 0.117, 0.891, 1.685, 3.198, 5]
y2 = [-0.092, 0.776, 1.524, 2.316, 3.089, 3.855]


plt.scatter(x, y1, marker='o', c='r')
plt.scatter(x, y2, color='blue')
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
