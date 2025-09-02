import pandas as pd

df = pd.read_csv('car_data.csv')
print(df.tail())
print(df.info())