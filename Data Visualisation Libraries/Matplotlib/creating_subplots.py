from matplotlib import pyplot
import pandas, numpy


x = numpy.linspace(0, 5, 11)
y = x**2

fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(10, 6))

axes[0].plot(x, y)
axes[1].plot(y, x)

pyplot.show()
