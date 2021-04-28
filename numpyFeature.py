import numpy as np

feature = np.arange(6, 21)
noise = np.random.random([15]) * 4
label = feature + noise
print(label)