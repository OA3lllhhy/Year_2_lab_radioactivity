import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('Al.csv')

# Extract the data
n = data.iloc[:, 0]
nstd = data.iloc[:,1]
lnn = data.iloc[:,2]
nd2 = data.iloc[:,3]
R = data.iloc[:,4]

lnnd2 = np.log(nd2)

# Plot the data of n vs R
plt.plot(R, lnnd2, 'o')
plt.show()
