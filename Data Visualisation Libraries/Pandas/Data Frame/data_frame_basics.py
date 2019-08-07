import pandas
from numpy.random import randn

df = pandas.DataFrame(randn(5,4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])

print(df)
print("===============")
print(df[['W', 'X']])

print("========Adding new col=======")
df['new'] = df['W'] + df['X']
print(df)

print("======= drop Tempory========")
df_new = df.drop('new', axis=1, inplace=False)
print(df_new)
print(df)

print("======= drop permanently========")
df.drop('new', axis=1, inplace=True)
df.drop('E', axis=0, inplace=True)
print(df)

print("=======Conditional slicing========")
print(df[(df['W'] > 0) & (df['X'] < 0)])

print("============reset index==============")
print(df.reset_index().drop('index', axis=1))
df['newIndex'] = 'AB BC CA XY'.split()
print(df.set_index('newIndex', inplace=False))
