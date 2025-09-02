import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('car_data.csv')

cols = df.select_dtypes(include='number').columns

if 'Car_name' in cols:
    cols = cols.drop('Car_name')

fig, axes = plt.subplots(nrows=2, ncols=(len(cols) + 1) // 2, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(cols):
    axes[i].hist(df[col])
    axes[i].set_title(col)

for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()