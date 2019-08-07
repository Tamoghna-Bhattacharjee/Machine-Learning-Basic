import numpy, pandas
from apyori import apriori


df = pandas.read_csv('Market_Basket.csv', header=None)
x = []

for i in range(7501):
    x.append([str(df.values[i, j]) for j in range(0, 20)])


rules = apriori(x, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

rules = list(rules)