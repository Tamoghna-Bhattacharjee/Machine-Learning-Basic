from matplotlib import pyplot
import pandas
import numpy


x = numpy.linspace(0, 5, 11)
y = x ** 2

pyplot.subplot(1,2,1)
pyplot.plot(x, y, 'r')
pyplot.xlabel("X - axis")
pyplot.ylabel("Y - axis")

pyplot.subplot(1,2,2)
pyplot.plot(y, x, 'g')
pyplot.xlabel("X - axis")
pyplot.ylabel("Y - axis")

pyplot.show()
