import pandas, numpy

outer = ['G1'] * 3 + ['G2'] * 3
inner = [1,2,3] * 2

heigher_level = pandas.MultiIndex.from_tuples(zip(outer, inner))

df = pandas.DataFrame(numpy.random.randn(6, 2), heigher_level, ['A', 'B'])


print("==============")
print(df)

print("===========look up============")
print(df.loc['G1'])
print(df.loc['G1'].loc[1, 'A'])

print("===========col level of index=======")
df.index.names = ['Group', 'Num']
print(df)

print("===========cross section==========")
print(df.xs(1, level="Num"))

