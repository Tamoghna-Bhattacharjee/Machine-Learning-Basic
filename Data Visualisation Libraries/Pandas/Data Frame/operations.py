import pandas, numpy


d = {'Col1': [1,2,3, 4], 'Col2': [444, 555, 666, 444], 'col3': ['a', 'b', 'c', 'd']}

df = pandas.DataFrame(d)

print(df)

print("========= unique =========")
print(df['Col2'].unique())

print("========= apply =========")
print(df['Col2'].apply(lambda x: x*2))

print("========= count =========")
print(df['Col2'].count())

print("========= get col and row indexes =========")
print("col index = ", df.columns)
print("row index = ", df.index)

print("========= sort =========")
print(df.sort_values(by='Col2'))