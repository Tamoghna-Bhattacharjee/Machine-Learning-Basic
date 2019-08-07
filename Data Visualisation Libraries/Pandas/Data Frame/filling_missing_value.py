import pandas, numpy

d = {'A': [1, numpy.nan, numpy.nan], 'B': [5,2,numpy.nan], 'C': [10,11,12]}

df = pandas.DataFrame(d)
print(df)

print("=========== dropna ===========")
print(df.dropna(axis=0), end="\n\n")
print(df.dropna(axis=1))

print("=========== fillna ============")
print(df.fillna("Fill"), end='\n\n')
print(df['B'].fillna(df['B'].mean()))

print("==================================")
company = ['Goog'] * 2 + ['FB'] * 2 + ['MSFT'] * 2
person = 'a b c d e f'.split()
sales = numpy.random.randint(100, 500, 6)

df = pandas.DataFrame(numpy.array([company, person, sales]).transpose(),
                      columns=['Company', 'Person', 'Sales'])

print(df)

gr = df.groupby(['Company'])


print("============= max ===========")
print(gr.max())

print("============= count ===========")
print(gr.count())


print("============= max ===========")
print(df.groupby(by=['Company']).describe())

