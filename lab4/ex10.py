import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('car_data.csv')

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Present_Price', y='Selling_Price', data=df)
plt.title('Relatia dintre Present_Price si Selling_Price')
plt.xlabel('Present_Price')
plt.ylabel('Selling_Price')
plt.grid(True)
plt.show()
