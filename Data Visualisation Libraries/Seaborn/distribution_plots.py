import seaborn as sns
from matplotlib import pyplot
import pandas, numpy


df = pandas.DataFrame(numpy.random.randn(100, 5), columns='A B C D E'.split())

# print(df)

# sns.distplot(df['A'], bins=20, kde=True)

# sns.jointplot(x='A', y='B', data=df, kind='hex')

# sns.pairplot(data=df)

# sns.rugplot(df['A'])


pyplot.show()
