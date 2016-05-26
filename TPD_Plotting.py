#!/usr/bin/env python
#coding=utf-8

'''
FT-IR: TPD plotting
'''

__author__ = "LI Kezhi" 
__date__ = "$2016-05-26$"
__version__ = "1.2.1"

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

###################################################

# Experiment data

position = "E:\\temp\\20160509_Mn1Fe9\\"

# Plotting parameters

differenceSpectra = False                   # If True, the first spectrum will be used as background
xHighRange, xLowRange = 1800, 1000           # "None" or a float
#xHighRange, xLowRange = None, None
minIntensity, maxIntensity = -0.015, 0.015  # "None" or a float
#minIntensity, maxIntensity = None, None
#color_choice = cm.jet
#color_choice = cm.RdBu_r
color_choice = cm.seismic

###################################################

# x, y grid

# Step 1: Obtain wavenumber step
filename = "1_NH3_ads\\series00120000.spa.csv"
count_x, start_x, end_x = 0, None, None
for line in open(position + filename, 'r'):
    try:
        if line != '#Converted with spa2csv tool\n':
            assert line != '\n'
            if start_x == None:
                start_x = float(line.split(',')[0])
            end_x = float(line.split(',')[0])
            count_x += 1
    except AssertionError:
        break
deltax = (start_x - end_x) / count_x
x = np.arange(end_x, start_x + deltax, deltax)
# Step 2: Generate delta_y
y = np.array([75, 100, 150, 200, 250, 300])
# Step 3: Generate the grid
X, Y = np.meshgrid(x, y)
Z = np.zeros([len(x), len(y)])


county = 0

# 75 Celcius Degree
filename = "2_75_des\\series00120036.spa.csv"
countx = 0
for line in open(position + filename, 'r'):
    try:
        if line != '#Converted with spa2csv tool\n':
            assert line != '\n'
            splitting = line.split(',')
            Z[countx, county] = float(splitting[1])
            if differenceSpectra == True and county != 0:
                Z[countx, county] -= Z[countx, 0]
            countx += 1
    except AssertionError:
        break
county += 1

# 100 Celsius Degree
filename = "3_100_C\\series00120018.spa.csv"
countx = 0
for line in open(position + filename, 'r'):
    try:
        if line != '#Converted with spa2csv tool\n':
            assert line != '\n'
            splitting = line.split(',')
            Z[countx, county] = float(splitting[1])
            if differenceSpectra == True and county != 0:
                Z[countx, county] -= Z[countx, 0]
            countx += 1
    except AssertionError:
        break
county += 1

# 150 Celsius Degree
filename = "4_150_C\\series00120018.spa.csv"
countx = 0
for line in open(position + filename, 'r'):
    try:
        if line != '#Converted with spa2csv tool\n':
            assert line != '\n'
            splitting = line.split(',')
            Z[countx, county] = float(splitting[1])
            if differenceSpectra == True and county != 0:
                Z[countx, county] -= Z[countx, 0]
            countx += 1
    except AssertionError:
        break
county += 1

# 200 Celsius Degree
filename = "5_200_C\\series00120018.spa.csv"
countx = 0
for line in open(position + filename, 'r'):
    try:
        if line != '#Converted with spa2csv tool\n':
            assert line != '\n'
            splitting = line.split(',')
            Z[countx, county] = float(splitting[1])
            if differenceSpectra == True and county != 0:
                Z[countx, county] -= Z[countx, 0]
            countx += 1
    except AssertionError:
        break
county += 1

# 250 Celsius Degree
filename = "6_250_C\\series00120018.spa.csv"
countx = 0
for line in open(position + filename, 'r'):
    try:
        if line != '#Converted with spa2csv tool\n':
            assert line != '\n'
            splitting = line.split(',')
            Z[countx, county] = float(splitting[1])
            if differenceSpectra == True and county != 0:
                Z[countx, county] -= Z[countx, 0]
            countx += 1
    except AssertionError:
        break
county += 1

# 300 Celsius Degree
filename = "7_300_C\\series00120018.spa.csv"
countx = 0
for line in open(position + filename, 'r'):
    try:
        if line != '#Converted with spa2csv tool\n':
            assert line != '\n'
            splitting = line.split(',')
            Z[countx, county] = float(splitting[1])
            if differenceSpectra == True and county != 0:
                Z[countx, county] -= Z[countx, 0]
            countx += 1
    except AssertionError:
        break
county += 1

if differenceSpectra == True:
    Z[:,0] = np.zeros_like(Z[:,0])

if xHighRange == None or xLowRange == None:    # Wavelength/cm^-1
    y_ = x[len(x)::-1]
else:
    x_low = int((xLowRange-x[0])/deltax)
    x_high = int((xHighRange-x[0])/deltax)
    y_ = x[x_high:x_low:-1]
x_ = y[len(y)::-1]       # Time/min
X_, Y_ = np.meshgrid(x_, y_)

ZT = np.transpose(Z)
ZT = np.fliplr(ZT)   # Left-right flip
if xHighRange == None or xLowRange == None:
    Z_ = ZT
else:
    Z_ = ZT[:, x_low:x_high]
Z_ = np.fliplr(Z_)   # Left-right flip
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
plt.ylabel("Temperature ($^o$C)")

cb = plt.colorbar()
cb.set_label('Kubelka-Munk')

plt.show()
