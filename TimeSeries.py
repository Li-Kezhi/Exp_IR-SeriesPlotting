#!/usr/bin/env python

'''
FT-IR: continuous spectra
Plotting 3D FT-IR spectra and integrate peaks of interests
Note: Integration program is based on Integration.py (v.1.0) written by me
'''

from __future__ import print_function

__author__ = "LI Kezhi"
__date__ = "$2017-06-24$"
__version__ = "2.0.1"

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import ticker

from lmfit.models import PolynomialModel
import mpltex

##### Parameters #####
initParams = {}
plotParams = {}

# Files loading
initParams['POSITION'] = '../1-NH3-ads-80/'  # Location of the files
initParams['SAMPLING_TIME'] = 0.03344952593  # Sampling time for each spectrum
initParams['PREFIX'] = 'series0001'  # eg. series00010000.csv

# Integration
initParams['IF_INTEGRATE'] = True  # Whether to integrate peaks of selection
initParams['INT_ZONES'] = [ 
    (1759, 1500), # COO-
    (1120, 950), # C-O
    (3030, 2870), # C-H
    (2385, 2350) # CO2
]  # Integration zones
initParams['INT_LABELS'] = [
    r'COO$^-$',
    'C-O',
    'C-H',
    r'CO$_2$'
]  # Labels

# Plotting
plotParams['DIFFRENCE_SPECTRA'] = False  # If True, the first spectrum will be used as background
# params['X_RANGE'] = (1700, 950)  # High and low range of x; THIS LINE CAN BE CANCELLED!
plotParams['INTENSITY_RANGE'] = (0, 0.0005)  # High and low range of intensity; THIS LINE CAN BE CANCELLED!
plotParams['COLOR'] = cm.seismic  # eg. cm.jet, cm.RdBu_r, cm.seismic, cm.hot, cm.CMRmap, cm.gnuplot2

##### End of Parameters #####


class Series(object):
    def __init__(self, **kwargs):
        '''
        Initiate by reading spectra files
        Input: **initParams
            Including data reading parameters and integration parameters
        Local variables:
            Matrix Z: x -> time in increasing direction, y-> wavelength in decreasing direction
            x, y: lists
                x -> wavelength in decreasing direction
                y -> time in increasing direction
            ifIntegrate: bool
            integration: integrated area
            intLabels: labels for the selected integration area
            position: location of the files
        '''
        # Experiment data
        POSITION = kwargs['POSITION']
        self.position = POSITION
        SAMPLING_TIME = kwargs['SAMPLING_TIME']

        # Integration
        intZones = kwargs['INT_ZONES']

        # Step 1: Obtain wavenumber step
        prefix = kwargs['PREFIX']
        filename = POSITION + prefix + '0000.spa.csv'
        data = np.loadtxt(filename, delimiter=',')
        self.x = np.transpose(data[:, 0])
        self.Z = np.zeros_like(data[:, 1])  # Initialize Z

        # Step 2: Read data
        county = 0

        if kwargs['IF_INTEGRATE'] is True:
            self.ifIntegrate = True
            self.integration = []  # Initialization of integration
            for i in xrange(len(intZones)):
                self.integration.append([])
            self.intLabels = kwargs['INT_LABELS']

        while True:
            try:
                suffix = "%04d.spa.csv" % county
                filename = POSITION + prefix + suffix
                data = np.loadtxt(filename, delimiter=',')
                self.y = data[:, 1]
                self.Z = np.vstack((self.Z, self.y))
                county += 1
                print('Manipulating ' + filename)

                # Integrate
                if kwargs['IF_INTEGRATE'] is True:
                    for i, zone in enumerate(intZones):
                        self.integration[i].append(self.intCurve(zone))

            except IOError:
                break
        self.Z = np.transpose(self.Z)
        self.Z = self.Z[:, 1:]

        # Step 3: Generate delta_y
        self.y = np.zeros_like(self.Z[0, :])
        for i in range(len(self.y)):
            self.y[i] = i * SAMPLING_TIME

    def intCurve(self, intZone):
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
        for i in xrange(np.size(self.x)):
            if self.x[i] <= startInt and startLine is None:
                startLine = i
            if startLine != None and self.x[i] <= endInt:
                endLine = i
                break
        x_bg = [self.x[startLine], self.x[endLine]]
        y_bg = [self.y[startLine], self.y[endLine]]

        bg_mod = PolynomialModel(1, prefix='bg_')   # Background
        pars = bg_mod.guess(y_bg, x=x_bg)

        mod = bg_mod

        init = mod.eval(pars, x=x_bg)
        out = mod.fit(y_bg, pars, x=x_bg)

        ##### Integration #####
        # Background subtraction
        comp = out.eval_components(x=self.x)
        out_param = out.params
        y_bg_fit = bg_mod.eval(params=out_param, x=self.x)
        y_bg_remove = self.y - y_bg_fit

        x_int = self.x[startLine:endLine]
        y_int = y_bg_remove[startLine:endLine]
        y_bg_fit_ = y_bg_fit[startLine:endLine]
        y_orig = self.y[startLine:endLine]

        integration = -np.trapz(y_int, x_int)

        return integration

    @mpltex.presentation_decorator
    def generalPlot(self, **kwargs):
        '''
        Plot figures
        Input: plotting paramaters - kwargs
        '''

        # Plotting parameters
        differenceSpectra = kwargs['DIFFRENCE_SPECTRA']
        xHighRange, xLowRange = kwargs.get('X_RANGE', (None, None))
        minIntensity, maxIntensity = kwargs.get('INTENSITY_RANGE', (None, None))
        color_choice = kwargs.get('COLOR', cm.seismic)


        ###################################################

        # Generate the grid
        if differenceSpectra is True:
            Z0 = self.Z[:, 0].copy()
            for i in range(np.shape(self.Z)[1]):
                self.Z[:, i] -= Z0

        if xHighRange is None or xLowRange is None:    # Wavelength/cm^-1
            y_ = self.x
        else:
            delta_x = self.x[1] - self.x[0]
            x_low = int((xLowRange-self.x[0])/delta_x)
            x_high = int((xHighRange-self.x[0])/delta_x)
            y_ = self.x[x_high:x_low]
        x_ = self.y[len(self.y)::-1]       # Time/min

        # Reshape the matrix
        ZT = np.transpose(self.Z)
        if xHighRange is None or xLowRange is None:
            Z_ = ZT
        else:
            Z_ = ZT[:, x_high:x_low]
        Z_ = np.flipud(Z_)   # Up-down flip

        # Plot
        if maxIntensity is None or minIntensity is None:
            im = plt.imshow(
                Z_,
                interpolation='bilinear',
                cmap=color_choice,
                aspect="auto",
                extent=[y_[0], y_[-1], x_[-1], x_[0]],
                vmax=Z_.max(), vmin=Z_.min()
            )
        else:
            im = plt.imshow(
                Z_,
                interpolation='bilinear',
                cmap=color_choice,
                aspect="auto",
                extent=[y_[0], y_[-1], x_[-1], x_[0]],
                vmax=maxIntensity, vmin=minIntensity
            )

        #plt.tight_layout()
        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("Time (min)")

        cb = plt.colorbar()
        cb.set_label(r'Kubelka-Munk ($\times$0.025)')
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        cb.ax.yaxis.set_major_formatter(formatter)

        plt.show()

        # Integration plotting
        if self.ifIntegrate is True:
            integration = self.integration
            fig, ax = plt.subplots()
            for i, singleSpecies in enumerate(integration):
                ax.plot(self.y, singleSpecies, label=self.intLabels[i])

            # ax.set_xlim(0, 90)

            ax.set_yticks([])
            # ax.tick_params(axis='x', top='off', bottom='off')

            ax.legend(loc='best')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Integration Area (A. U.)')

            plt.show()

        ##### Text output #####
        if self.ifIntegrate is True:
            result_txt = self.position + 'integration.txt'
            headLine = 'Time(min)'
            for i in xrange(len(integration)):
                headLine += '   '
                headLine += repr(self.intLabels[i])
            integration = np.fastCopyAndTranspose(integration)
            self.y = np.array(self.y, ndmin=2)
            self.y = np.transpose(self.y)
            integration = np.hstack((self.y, integration))

            np.savetxt(result_txt, integration, fmt='%.3e', header=headLine)


if __name__ == '__main__':
    testData = Series(**initParams)
    testData.generalPlot(**plotParams)
