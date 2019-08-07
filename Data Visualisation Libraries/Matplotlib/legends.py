from matplotlib import pyplot
import pandas, numpy

x = numpy.linspace(0, 5, 11)

fig = pyplot.figure(figsize=(8, 5))
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, x**2, label='square')
axes.plot(x, x**3, label= 'cube')

axes.legend(loc=(0.2,0.3))
pyplot.show()