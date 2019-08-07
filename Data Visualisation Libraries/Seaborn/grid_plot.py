import seaborn as sns
from matplotlib import pyplot
import pandas, numpy


df = pandas.DataFrame(numpy.random.randn(100, 4), columns='A B C D'.split())


grid = sns.PairGrid(df)
# grid.map(sns.scatterplot)
grid.map_diag(sns.distplot)
grid.map_upper(sns.scatterplot)
grid.map_lower(sns.kdeplot)

pyplot.show()