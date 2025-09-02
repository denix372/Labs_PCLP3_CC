import pandas as pd
import numpy as np

def calculate_statistics(filename):
    """
    Calculează media, valoarea maximă, valoarea minimă și deviația standard a datelor.
    """
    # TODO
    df = pd.read_csv(filename)
    numeric_df = df.select_dtypes(include=np.number)

    statistics = {
        'Mean': numeric_df.mean(),
        'Max': numeric_df.max(),
        'Min': numeric_df.min(),
        'Standard Deviation': numeric_df.std()
    }

    return statistics

filename = 'grades.csv'
# Calculează statisticile
statistics = calculate_statistics(filename)
# Afișează statisticile
print("Statistici pentru note:")
for stat, value in statistics.items():
    print(f"{stat}:\n{value}")