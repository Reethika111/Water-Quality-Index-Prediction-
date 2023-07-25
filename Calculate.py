import pandas as pd
import numpy as np

dataset = pd.read_csv('WQI.csv')
weightage_factors = np.array([0.005, 0.004, 0.005, 0.2, 0.117647059, 0.05, 0.002, 0.004], dtype=object)
wi = np.array([0.012898, 0.010319, 0.012898, 0.515933,  0.30349, 0.128983, 0.005159, 0.010319], dtype=object)
sn = np.array([200,250,200,5,8.5,20,500,250], dtype=object)

result = (dataset.values /sn) * 100
print(result)

wqi_values = np.sum(result, axis=1)

dataset['WQI'] = wqi_values

