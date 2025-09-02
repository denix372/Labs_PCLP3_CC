import pandas as pd

df = pd.read_csv('housePrices.csv')
max_price = df['price'].max()
print( df[ df['price'] == max_price ])