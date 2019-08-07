import seaborn as sns
import numpy, pandas
from matplotlib import pyplot

d = {'total_bill': [16.99, 10.34, 21.01, 23.68, 24.59], 'tip': [1.01, 1.666, 3.50, 3.31, 3.61],
     'sex': ['Female', 'Male', 'Male', 'Male', 'Female'], 'smoker': ['No', 'No', 'No', 'No', 'Yes'],
     'day': ['sun']*2 + ['mon']*3, 'time': ['Dinner']*5, 'size': [2,3,3,2,4]}

df = pandas.DataFrame(d)

# sns.barplot(x='sex', y='total_bill', data=df, estimator=numpy.std)

# sns.boxplot(x='day', y='total_bill', data=df, hue='smoker')

# sns.violinplot(x='day', y='total_bill', data=df)
# sns.swarmplot(x='day', y='total_bill', data=df, color='black')

sns.catplot(x='day', y='total_bill', data=df, kind='box')
pyplot.show()