import pandas as pd

df = pd.read_csv('housePrices.csv')
df_filtrat = df[df['area'] > 100]
print(df_filtrat)