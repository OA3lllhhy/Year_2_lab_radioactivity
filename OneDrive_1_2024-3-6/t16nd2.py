import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.stats as stats
from scipy import integrate
import scipy.optimize as op
from scipy.stats import chi2

# Folder path
folder_path1 = 'task16/T1'
folder_path2 = 'task16/T2'
folder_path3 = 'task16/T3'

# Read all data from the folder
data1 = []
distance = []
for file in os.listdir(folder_path1):
    if file.endswith('.csv'):
        data1.append(pd.read_csv(folder_path1 + '/' + file))
        distance.append(float(file[:-4]))

data2 = []
for file in os.listdir(folder_path2):
    if file.endswith('.csv'):
        data2.append(pd.read_csv(folder_path2 + '/' + file))

data3 = []
for file in os.listdir(folder_path3):
    if file.endswith('.csv'):
        data3.append(pd.read_csv(folder_path3 + '/' + file))

# Merge the data from the three folders according to the distance
# data  = list(zip(data1, data2, data3))
data = data1

d = np.array(distance)/100
# print(d)

n = []
for i in range(len(data)):
    n.append((float(data[i].sum())))
# print(n)

std = []
for i in range(len(data)):
    std.append((float(data[i].std())))
# print(std)

t = []
for i in range(len(data)):
    t.append(len(data[i]))

# calculate the count rate
# n_t = []
# for i in range(len(t)):
#     n_t.append(n[i]/t[i])

U = []
for i in range(len(data)):
    U.append((n[i]/t[i])*d[i]**2)

# Calculate the fraction lost due to the dead time of the detector
def fraction_lost(t, n, t_d):
    return np.exp(-(n/t)* t_d)

n_dt = []
for i in range(len(data)):
    n_dt.append(U[i]/fraction_lost(t[i], n[i], 0.0000002))



error = []
for i in range(len(data)):
    # error.append(U[i] * np.sqrt((2*0.003/d[i])**2 + (np.array(std[i])/n[i])**2))
    error.append(U[i] * np.sqrt((2*0.003/d[i])**2 + (1/(n[i]))))

# sort d and then U according to d
d, U, error = zip(*sorted(zip(d, U, error)))
d = np.array(d)
U = np.array(U)
# print(U)
error = np.array(error)


# calculate the non point source effect 
def cylinder(p, r, a):
    # p radial coordinate centered at the source
    # r perpendicular distance from the source to the detector
    # a the side length of the detector
    return (1/np.pi) *  np.arctan(a/2 * r/(r**2 + p**2)) * 1/np.sqrt(1 + (2/a * (r**2 + p**2)/r)**2) * p * 2*np.pi

def cs(d, A): # predict count rate of source at distance x (A: activity)
    source_diameter = 0.020
    detector_length = 0.015
    #rho and width correspond to source_diameter and detector_length respectively
    return [A/(np.pi*source_diameter**2/4) * integrate.quad(cylinder, 0, source_diameter/2, args=(i, detector_length))[0] for i in d] * d**2



# Fitting 
def exp(x, A, k):
    return A * np.exp(-x/k)

def linear(x, m, A):
    return m*x + A

att_fit, att_cov = op.curve_fit(linear, d[6:], U[6:], p0=[0, 15]) # attenuation , sigma=3*error[6:]*np.sqrt((1+0.25/d[6:]**2))
nps_fit, _ = op.curve_fit(lambda x, A: cs(x, A) + exp(x, *att_fit) - exp(0, *att_fit), d[:8], U[:8], p0=[8e5]) # non-point source

# Doing linear regression for selected data points
slope7, intercept7, r_value7, p_value7, intercept_stderr7 = stats.linregress(d[6:], U[6:])
# Calculate the chi squared value
chisq = np.sum((U[6:] - (slope7*d[6:] + intercept7))**2/error[6:]**2)
# calculate the p value
p = 1 - stats.chi2.cdf(chisq, len(d[6:])-2)

# calculate the R squared value
r2 = 1 - (np.sum((U[6:] - (slope7*d[6:] + intercept7))**2)/np.sum((U[6:] - np.mean(U[6:]))**2))
print('R squared value:', r2)



# print the results with 4 decimal places
print(f'For distance 7 onwards: slope = {slope7:.4f}, intercept = {intercept7:.4f}, r2_value = {r_value7**2:.4f}, p_value = {p_value7}, intercept_stderr = {intercept_stderr7:.4f}')
# slope = []
# intercept = []
# r_value = []
# p_value = []
# intercept_stderr = []
# for i in range(len(d)):
#     slope_, intercept_, r_value_, p_value_, intercept_stderr_ = stats.linregress(d[i:], U[i:])
#     slope.append(float(slope_))
#     intercept.append(float(intercept_))
#     r_value.append(float(r_value_))
#     p_value.append(float(p_value_))
#     intercept_stderr.append(float(intercept_stderr_))

# create a csv file to store the linear regression results with 4 decimal places
# df = pd.DataFrame({'distance': d, 'slope': slope, 'intercept': intercept, 'r2_value': r_value**2, 'p_value': p_value, 'intercept_stderr': intercept_stderr})
# df.to_csv('task16/linear_regression.csv', index=False)

# for i in range(len(d)):
#     print(f'For distance {i} onwards: slope = {slope[i]}, intercept = {intercept[i]}, r2_value = {r_value[i]**2}, p_value = {p_value[i]}, intercept_stderr = {intercept_stderr[i]}')


x = np.linspace(0, 1, 100)
d_fit = np.linspace(0, 1, 100)
trans_nps = np.linspace(100, 100, 100)

# plot U against distance
plt.figure(figsize=(8, 5), dpi=100)
# plt.plot(d_fit, cs(d_fit, 15), color = 'black', label = 'Non point source effect')
# plt.plot(d, n_dt, 'o', label = 'Data points with dead time correction')
plt.plot(x, cs(x,*nps_fit), label='Non-point Source Effect', color='blue')
plt.plot(x, cs(trans_nps, *nps_fit) - cs(x,*nps_fit), '-', color='orange', label='Non-point Source Effect significance')
plt.plot(x, slope7*x + intercept7, color = 'red', label = f'Attenuation Effect: y = {slope7:.2f}x + {intercept7:.2f}, $R^2$ = {r_value7**2:.4f}')
plt.errorbar(d, U, yerr=error, capsize = 3 , fmt='o', color = 'grey', alpha = 0.9)
plt.xlabel('Distance (m)')
plt.ylabel('$U = n d^2$')
plt.xlim(0, 1)
# plt.ylim(0, 20)
# plt.xscale('log')
plt.legend(loc='best')
plt.grid()
plt.title('U against distance')
plt.show()