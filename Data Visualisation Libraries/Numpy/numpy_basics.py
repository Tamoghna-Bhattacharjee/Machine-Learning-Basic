import numpy

print("=======Creating array===========")
print(numpy.array([1,2,3]))

print("=======numpy Range(arange)===========")
print(numpy.arange(0,10,2))

print("=======default initialization===========")
print("1-d:\n", numpy.zeros(5))
print("2-d:\n", numpy.zeros((3,3)))
print("1-d:\n", numpy.ones(5))
print("2-d:\n", numpy.ones((3,3)))

print("=======linear spacing===========")
print(numpy.linspace(0, 5, 10))

print("=======identity matrix===========")
print(numpy.eye(3))

print("=======random.rand===========")
print(numpy.random.rand(5))
print(numpy.random.rand(5, 5))

print("=======random.randint===========")
print(numpy.random.randint(0, 100))
print(numpy.random.randint(0, 100, 10))

print("=======random.randn  => normal distribution===========")
print(numpy.random.randn(5), "\n")
print(numpy.random.randn(5, 5))

print("=======reshape===========")
arr = numpy.arange(25)
arr = arr.reshape(5,5)
print(arr)
print("shape = ", arr.shape)
print("data type = ", arr.dtype)

print(f"max = {arr.max()} \t min = {arr.min()}")
print(f"max index = {arr.argmax()} \t min index = {arr.argmin()}")

print("=======arrays detail===========")
arr = numpy.arange(11)
slice_arr = arr[:5].copy()
print("slice array = ", slice_arr)
slice_arr[:] = 99
print("slice array = ", slice_arr)
print("original array = ", arr)

print("=======2d-arrays detail===========")
arr = numpy.arange(50).reshape(5,10)
print(arr)
print(arr[0:5, 5:10])
print(arr[arr > 25])
print("sum = ", arr.sum())

