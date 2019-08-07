import pandas, numpy

d1 = {'A': 'A0 A1 A2'.split(), 'B': 'B0 B1 B2'.split()}
d2 = {'C': 'C0 C1 C2'.split(), 'D': 'D0 D1 D2'.split()}

df1 = pandas.DataFrame(d1)
df2 = pandas.DataFrame(d2)

print(df1)
print(df2)

print("========== concat axis = 1 ============")
print(pandas.concat([df1, df2], axis=1))

print("=====================")
df1['K'] = 'K0 K1 K2'.split()
df2['K'] = 'K0 K1 K2'.split()

print(df1)
print(df2)

print("========== mering ===========")
print(pandas.merge(df1, df2, on='K'))


print("=====================")
df1.set_index(keys='K', inplace=True)
df2.set_index('K', inplace=True)

print(df1)
print(df2)

print("========== joining ===========")
print(df1.join(df2))

