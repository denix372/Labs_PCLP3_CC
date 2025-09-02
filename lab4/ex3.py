import pandas as pd

df = pd.read_csv('housePrices.csv')
print(df['price'].mean())
print(df['price'].max())
print(df['price'].min())
print(df['price'].std())