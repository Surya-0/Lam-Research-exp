import numpy as np
import pandas as pd

# Generate a time series dataset
def generate_time_series_data(n, start='2020-01-01'):
    date_rng = pd.date_range(start=start, periods=n, freq='D')
    df = pd.DataFrame(date_rng, columns=['date'])
    df['value'] = np.sin(np.linspace(0, 20, n)) + np.random.normal(scale=0.5, size=n)
    df.set_index('date', inplace=True)
    return df

# Example usage
n = 36  # Number of days
df = generate_time_series_data(n)
print(df)