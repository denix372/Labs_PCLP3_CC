import pandas as pd

df = pd.read_csv('housePrices.csv')
max_price = df['price'].max()
max_area = df['area'].max()

df['price'] = df['price']/max_price
df['area'] = df['area']/max_area

df.to_csv('housePrices_normalized.csv', index=False)