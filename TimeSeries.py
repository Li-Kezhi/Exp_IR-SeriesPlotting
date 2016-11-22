#!/usr/bin/env python

'''
FT-IR: continuous spectra
'''

__author__ = "LI Kezhi" 
__date__ = "$2016-11-22$"
__version__ = "1.2.2"

import numpy as np
import matplotlib.cm as cm
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

###################################################

# Experiment data

position = "./20161121_test/2_NO_cycle_2min/spectra/"
SAMPLING_TIME = 0.033432280262746  # Sampling time for each spectrum

# Plotting parameters

differenceSpectra = True                   # If True, the first spectrum will be used as background
# xHighRange, xLowRange = 1800, 1000           # "None" or a float
xHighRange, xLowRange = None, None
minIntensity, maxIntensity = -0.1, 0.1  # "None" or a float
# minIntensity, maxIntensity = None, None
#color_choice = cm.jet
#color_choice = cm.RdBu_r
color_choice = cm.seismic

###################################################

# x, y grid

# Step 1: Obtain wavenumber step
filename = position + "series0000.spa.csv"
data = np.loadtxt(filename, delimiter = ',')
x = np.transpose(data[:, 0])
delta_x = (x[-1] - x[0]) / (len(x) - 1)

Z = np.zeros_like(data[:, 1])  # Initialize Z

# Step 2: Read data
county = 0
prefix = 'series'


while True:
    try:
        suffix = "%04d.spa.csv" % county
        filename = position + prefix + suffix
        data = np.loadtxt(filename, delimiter=',')
        y = data[:, 1]
        Z = np.vstack((Z, y))        
        county += 1
        print ('Manipulating' + filename)
    except IOError:
        break
Z = np.transpose(Z)
Z = Z[:, 1:]

# Step 3: Generate delta_y
y = np.zeros_like(Z[:,0])
for i in range(len(y)):
    y[i] = i * SAMPLING_TIME
delta_y = SAMPLING_TIME

# Step 4: Generate the grid
X, Y = np.meshgrid(x, y)

if differenceSpectra == True:
    Z0 = Z[:,0].copy()
    for i in range(np.shape(Z)[1]):
        Z[:,i] -= Z0
    # Z[:,0] = np.zeros_like(Z[:,0])


if xHighRange == None or xLowRange == None:    # Wavelength/cm^-1
    y_ = x
else:
    x_low = int((xLowRange-x[0])/delta_x)
    x_high = int((xHighRange-x[0])/delta_x)
    y_ = x[x_high:x_low]
x_ = y[len(y)::-1]       # Time/min
X_, Y_ = np.meshgrid(x_, y_)

ZT = np.transpose(Z)
if xHighRange == None or xLowRange == None:
    Z_ = ZT
else:
    Z_ = ZT[:, x_high:x_low]
Z_ = np.flipud(Z_)   # Up-down flip

if maxIntensity == None or minIntensity == None:
    im = plt.imshow(Z_, 
                    interpolation='bilinear', 
                    cmap=color_choice,
                    aspect="auto",
                    extent=[y_[0], y_[-1], x_[-1], x_[0]],
                    vmax=Z_.max(), vmin=Z_.min())
else:
    im = plt.imshow(Z_, 
                    interpolation='bilinear', 
                    cmap=color_choice,
                    aspect="auto",
                    extent=[y_[0], y_[-1], x_[-1], x_[0]],
                    vmax=maxIntensity, vmin=minIntensity)

#plt.tight_layout()
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Time (min)")

cb = plt.colorbar()
cb.set_label('Kubelka-Munk')

plt.show()
