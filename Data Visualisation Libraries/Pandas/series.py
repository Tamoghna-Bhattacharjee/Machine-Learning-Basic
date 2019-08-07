import pandas
import numpy

data = [10,20,30]
arr = numpy.array(data)
d = {'a': 10, 'b': 20, 'c': 30}
label = ['a', 'b', 'c']

print("===============")
s1 = pandas.Series(data)
print(s1)

print("===============")
s2 = pandas.Series(arr)
print(s2)

print("===============")
s3 = pandas.Series(arr, label)
print(s3)

print("===============")
s4 = pandas.Series(d)
print(s4)

print("===============")

s5 = pandas.Series(data, ['a', 'b', 'c'])
s6 = pandas.Series(arr*3, ['a', 'b', 'x'])
print(s5)
print(s6)

print("s5 + s6")
print(s5 + s6)
