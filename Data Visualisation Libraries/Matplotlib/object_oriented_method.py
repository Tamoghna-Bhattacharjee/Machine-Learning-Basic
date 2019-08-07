from matplotlib import pyplot
import pandas, numpy


x = numpy.linspace(0, 5, 11)
y = x**2

fig = pyplot.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.3, 0.3])

axes1.plot(x, y)
axes1.set_xlabel("X-axis")
axes1.set_ylabel("Y-axis")

axes2.plot(y, x)
axes2.set_xlabel("X-axis")
axes2.set_xlabel("Y-axis")

# fig.savefig("My_matplotlib_graph.png")

pyplot.show()
