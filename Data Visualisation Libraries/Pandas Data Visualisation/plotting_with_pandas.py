import numpy, pandas
from matplotlib import pyplot

df1 = pandas.read_csv('df1', index_col=0)
df2 = pandas.read_csv('df2')

# graph = df1['A'].hist(bins=30)
# df2['a'].plot(kind='area', alpha=0.1)

# df2.plot(kind='bar', stacked=True)

# df1.plot.line(x='A', y='B')

df1.plot.scatter(x='A', y='B', c='C')

pyplot.show()
