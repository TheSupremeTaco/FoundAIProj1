import numpy as np

with open("Credit_N400_p9.csv") as file_name:
    array = np.loadtxt(file_name, delimiter=",")

print(array)

import numpy as np


with open("Credit_N400_p9.csv") as file_name:
    array = np.genfromtxt(file_name, dtype=None, names=True, delimiter=",")

print(array)
for i in range(len(array)):
    if array[i][6] == b'Male':
        array[i][6] = 0
print(array[2][6])
