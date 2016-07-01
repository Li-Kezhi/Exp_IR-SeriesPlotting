#!/usr/bin/env python
#coding=utf-8

'''
HPLC: 3d plotting
'''

__author__ = "LI Kezhi" 
__date__ = "$2016-07-01$"
__version__ = "1.2.1"

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

###################################################

# Experiment data

position = "E:\\SkyDrive\\Sharing data\\HPLC\\20160622\\20160622\\"

# Plotting parameters

xHighRange, xLowRange = 8, 0           # "None" or a float
#xHighRange, xLowRange = None, None
minIntensity, maxIntensity = 0, 0.02  # "None" or a float
#minIntensity, maxIntensity = None, None
#color_choice = cm.jet
#color_choice = cm.RdBu_r
#color_choice = cm.seismic
color_choice = cm.hot

###################################################

# x, y grid

# Step 1: Obtain wavelength step (y)
filename = "Benzaldehyde1209.arw"
count_y, start_y, end_y = 0, None, None
for file in open(position + filename, 'r'):
    lines = file.split('\r')
    for line in lines:
        if "\"" not in line:
            splitting = line.expandtabs(1).split()
            if start_y == None:
                start_y = float(splitting[1])  # Filter: omit the first comment
            if end_y == None:
                end_y = float(splitting[-1])
            count_y = len(splitting) - 1
            break
deltay = (start_y - end_y) / (count_y - 1)
y = np.arange(end_y, start_y + deltay, deltay)

# Step 2: Generate time step (x)
count_x, start_x, end_x = 0, None, None
for file in open(position + filename, 'r'):
    lines = file.split('\r')
    start_flag = False
    for line in lines:
        if not line == '':
            splitting = line.expandtabs(1).split()
            if splitting[0] == '0':
                start_flag = True
            if start_flag == True:
                if start_x == None:
                    start_x = float(splitting[0])  # Filter: omit the first comment
                end_x = float(splitting[0])
                count_x += 1
deltax = (-start_x + end_x) / (count_x - 1)
x = np.arange(start_x, end_x + deltax, deltax)

# Step 3: Generate the grid
X, Y = np.meshgrid(x, y)
Z = np.zeros([len(x), len(y)])

a = Z.shape

county = 0

# Step 4: Data import
for file in open(position + filename, 'r'):
    lines = file.split('\r')
    start_flag = False
    countx = 0
    for line in lines:
        try:
            if not line == '':
                splitting = line.expandtabs(1).split()
                if splitting[0] == '0':
                    start_flag = True
                if start_flag == True:
                    Z[countx, :] = splitting[1:]

            countx += 1
        except IndexError:                   # Unknown index error
            break


if xHighRange == None or xLowRange == None:    # Wavelength/cm^-1
    y_ = x[len(x)::-1]
else:
    x_low = int((xLowRange-x[0])/deltax)
    x_high = int((xHighRange-x[0])/deltax)
    y_ = x[x_high:x_low:-1]
x_ = y       # Time/min  BUG
X_, Y_ = np.meshgrid(x_, y_)

ZT = np.transpose(Z)
ZT = np.fliplr(ZT)   # Left-right flip
if xHighRange == None or xLowRange == None:
    Z_ = ZT
else:
    Z_ = ZT[:, len(x)-x_high:len(x)-x_low]
Z_ = np.fliplr(Z_)   # Left-right flip
Z_ = np.flipud(Z_)   # Up-down flip

if maxIntensity == None or minIntensity == None:
    im = plt.imshow(Z_, 
                    interpolation='bilinear', 
                    cmap=color_choice,
                    aspect="auto",
                    extent=[y_[-1], y_[0], x_[-1], x_[0]],
                    vmax=Z_.max(), vmin=Z_.min())
else:
    im = plt.imshow(Z_, 
                    interpolation='bilinear', 
                    cmap=color_choice,
                    aspect="auto",
                    extent=[y_[-1], y_[0], x_[-1], x_[0]],
                    vmax=maxIntensity, vmin=minIntensity)

#plt.tight_layout()
plt.xlabel("Time (min)")
plt.ylabel("Wavelength (nm)")

cb = plt.colorbar()
#cb.set_label('Kubelka-Munk')

plt.show()
