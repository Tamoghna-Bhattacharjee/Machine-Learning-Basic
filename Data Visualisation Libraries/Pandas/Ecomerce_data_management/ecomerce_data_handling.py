import pandas, numpy

"""
['Address', 'Lot', 'AM or PM', 'Browser Info', 'Company', 'Credit Card',
       'CC Exp Date', 'CC Security Code', 'CC Provider', 'Email', 'Job',
       'IP Address', 'Language', 'Purchase Price'],
"""

df = pandas.read_csv('Ecommerce_Purchases.csv')

print("average Purchase price = ", df['Purchase Price'].mean())

print("highest Purchase price = ", df['Purchase Price'].max())

print("lowest Purchase price = ", df['Purchase Price'].min())

print('People with english language', len(df[df['Language'] == 'en']))

print("number of lower people = ", df[df['Job'] == 'Lawyer']['Job'].count())

print("5 most common job titles\n", df['Job'].value_counts().head(5))

print("Specific perchase", df[df['Lot'] == '90 WT']['Purchase Price'])