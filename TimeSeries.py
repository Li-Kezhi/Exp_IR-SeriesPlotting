#!/usr/bin/env python

'''
FT-IR: continuous spectra
Plotting 3D FT-IR spectra and integrate peaks of interests
Note: Integration program is based on Integration.py (v.1.0) written by me
'''

__author__ = "LI Kezhi"
__date__ = "$2017.01.05$"
__version__ = "1.4.0"

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import ticker

from lmfit.models import PolynomialModel
import mpltex


@mpltex.presentation_decorator
def plot():
    # Experiment data

    position = "./1-CH3OH-ads/"
    SAMPLING_TIME = 0.521748571428571  # Sampling time for each spectrum

    # Plotting parameters

    differenceSpectra = False  # If True, the first spectrum will be used as background
    xHighRange, xLowRange = 1700, 950  # "None" or a float
    # xHighRange, xLowRange = None, None
    minIntensity, maxIntensity = 0, 0.025  # "None" or a float
    # minIntensity, maxIntensity = None, None
    # color_choice = cm.jet
    # color_choice = cm.RdBu_r
    color_choice = cm.seismic
    # color_choice = cm.hot
    # color_choice = cm.CMRmap
    # color_choice = cm.gnuplot2

    # Options

    # Integration
    intZones = [              # Integration zones
        (1759, 1500), # COO-
        (1120, 950), # C-O
        (3030, 2870), # C-H
        (2385, 2350) # CO2
    ]
    intLabels = [
        r'COO$^-$',
        'C-O',
        'C-H',
        r'CO$_2$'
    ]
    ########## Integration ##########
    def intCurve(x, y, intZone):
        '''
        Integrate the area in a given zone.
        Input:
            x, y - arrays, original data
            intZones - tuple, eg. (3000, 2700)
        '''
        startInt = intZone[0]   # Head x point; integration range
        endInt = intZone[1]

        ##### Background #####
        startLine, endLine = None, None
        for i in xrange(np.size(x)):
            if x[i] <= startInt and startLine is None:
                startLine = i
            if startLine != None and x[i] <= endInt:
                endLine = i
                break
        x_bg = [x[startLine], x[endLine]]
        y_bg = [y[startLine], y[endLine]]

        bg_mod = PolynomialModel(1, prefix='bg_')   # Background
        pars = bg_mod.guess(y_bg, x=x_bg)

        mod = bg_mod

        init = mod.eval(pars, x=x_bg)
        out = mod.fit(y_bg, pars, x=x_bg)

        ##### Integration #####
        # Background subtraction
        comp = out.eval_components(x=x)
        out_param = out.params
        y_bg_fit = bg_mod.eval(params=out_param, x=x)
        y_bg_remove = y - y_bg_fit

        x_int = x[startLine:endLine]
        y_int = y_bg_remove[startLine:endLine]
        y_bg_fit_ = y_bg_fit[startLine:endLine]
        y_orig = y[startLine:endLine]

        integration = -np.trapz(y_int, x_int)

        # print(str(integration[i]))

        # # Plotting
        # if county == 50:
        #     plt.plot(x, y, 'b.')
        #     plt.plot(x_bg, out.best_fit, 'r-')    # Background plotting
        #     plt.xlim([x[0], x[-1]])
        #     plt.fill_between(x_int, y_orig, y_bg_fit_, facecolor='green')
        #     plt.show()

        return integration

    ###################################################

    # x, y grid

    # Step 1: Obtain wavenumber step
    filename = position + "series00120000.spa.csv"
    data = np.loadtxt(filename, delimiter=',')
    x = np.transpose(data[:, 0])
    delta_x = (x[-1] - x[0]) / (len(x) - 1)

    Z = np.zeros_like(data[:, 1])  # Initialize Z

    # Step 2: Read data
    county = 0
    prefix = 'series0012'

    integration = []  # Initialization of integration
    for i in xrange(len(intZones)):
        integration.append([])

    while True:
        try:
            suffix = "%04d.spa.csv" % county
            filename = position + prefix + suffix
            data = np.loadtxt(filename, delimiter=',')
            y = data[:, 1]
            Z = np.vstack((Z, y))
            county += 1
            print ('Manipulating ' + filename)

            # Integrate
            for i, zone in enumerate(intZones):
                integration[i].append(intCurve(x, y, zone))

        except IOError:
            break
    Z = np.transpose(Z)
    Z = Z[:, 1:]

    # Step 3: Generate delta_y
    y = np.zeros_like(Z[0,:])
    for i in range(len(y)):
        y[i] = i * SAMPLING_TIME
    delta_y = SAMPLING_TIME

    # Step 4: Generate the grid
    X, Y = np.meshgrid(x, y)

    if differenceSpectra is True:
        Z0 = Z[:, 0].copy()
        for i in range(np.shape(Z)[1]):
            Z[:, i] -= Z0
        # Z[:, 0] = np.zeros_like(Z[:,0])


    if xHighRange is None or xLowRange is None:    # Wavelength/cm^-1
        y_ = x
    else:
        x_low = int((xLowRange-x[0])/delta_x)
        x_high = int((xHighRange-x[0])/delta_x)
        y_ = x[x_high:x_low]
    x_ = y[len(y)::-1]       # Time/min
    X_, Y_ = np.meshgrid(x_, y_)

    ZT = np.transpose(Z)
    if xHighRange is None or xLowRange is None:
        Z_ = ZT
    else:
        Z_ = ZT[:, x_high:x_low]
    Z_ = np.flipud(Z_)   # Up-down flip

    if maxIntensity is None or minIntensity is None:
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
    cb.set_label(r'Kubelka-Munk ($\times$0.025)')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    cb.ax.yaxis.set_major_formatter(formatter)

    plt.show()

    # Integration plotting
    fig, ax = plt.subplots()
    for i, singleSpecies in enumerate(integration):
        ax.plot(y, singleSpecies, label=intLabels[i])
    
    ax.set_xlim(0, 90)

    ax.set_yticks([])
    # ax.tick_params(axis='x', top='off', bottom='off')

    ax.legend(loc='best')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Integration Area (A. U.)')

    plt.show()

    ##### Text output #####
    result_txt = position + 'integration.txt'
    headLine = 'Time(min)'
    for i in xrange(len(integration)):
        headLine += '   '
        headLine += repr(intLabels[i])
    integration = np.fastCopyAndTranspose(integration)
    y = np.array(y, ndmin=2)
    y = np.transpose(y)
    integration = np.hstack((y, integration))

    np.savetxt(result_txt, integration, fmt='%.3e', header=headLine)

plot()
