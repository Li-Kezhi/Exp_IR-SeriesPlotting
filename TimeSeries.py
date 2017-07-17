#!/usr/bin/env python

'''
FT-IR: continuous spectra
Plotting 3D FT-IR spectra and integrate peaks of interests
Note: Integration program is based on Integration.py (v.1.0) written by me
'''

from __future__ import print_function

__author__ = "LI Kezhi"
__date__ = "$2017-06-30$"
__version__ = "2.0.4"

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import ticker

from lmfit.models import PolynomialModel
import mpltex

##### Parameters #####
initParams = {}
plotParams = {}
FourierAnalysisParams = {}

# Files loading
initParams['POSITION'] = '../4-VOacac2-1min/'  # Location of the files
initParams['SAMPLING_TIME'] = 0.03344952593  # Sampling time for each spectrum
initParams['PREFIX'] = 'series0014'  # eg. series00010000.csv

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
# plotParams['INTENSITY_RANGE'] = (0, 0.0005)  # High and low range of intensity; THIS LINE CAN BE CANCELLED!
plotParams['COLOR'] = cm.seismic  # eg. cm.jet, cm.RdBu_r, cm.seismic, cm.hot, cm.CMRmap, cm.gnuplot2

# Fourier analysis parameters
# FourierAnalysisParams['X_RANGE'] = (1700, 950)  # High and low range of x; THIS LINE CAN BE CANCELLED!
FourierAnalysisParams['TIME_RANGE'] = (85, 105)  # Analysis time range; THIS LINE CAN BE CANCELLED!
FourierAnalysisParams['REPEAT_CYCLE'] = 10  # Repeat cycles; THIS LINE CAN BE CANCELLED!
FourierAnalysisParams['IF_PLOT_PHASE_ANGLE'] = True
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
        else:
            self.ifIntegrate = False

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
    def FourierTransformPlot(self, **kwargs):
        '''
        Fourier transform of the original time-related matrix
        Input:
            repeatCycle: 1 for non-periodic (default), and > 1 for the repeat cycles
            kwargs:
                X_RANGE: plotting range, default - (4000, 700)
        Output:
            TXT report and plotting
        '''
        repeatCycle = kwargs.get('REPEAT_CYCLE', 1)
        X_RANGE = kwargs.get('X_RANGE', (4000, 700))
        TIME_RANGE = kwargs.get('TIME_RANGE', (self.y[0], self.y[-1]))
        delta_y = self.y[1] - self.y[0]
        timeIndex = (int(TIME_RANGE[0]/delta_y), int(TIME_RANGE[1]/delta_y))
        Z_fft = np.fft.rfft(self.Z[:, timeIndex[0]:timeIndex[1]]) / self.Z.shape[1]
        wavenumber = self.x
        amplitude = np.abs(Z_fft[:, repeatCycle])
        phaseAngle = np.angle(Z_fft[:, repeatCycle])
        zeroAmplitude = np.abs(Z_fft[:, 0])
        # Write data
        result_txt = self.position + 'FourierTransform.txt'
        if kwargs['IF_PLOT_PHASE_ANGLE'] == True:
            headLine = 'Wavenumber(cm-1)   Background   Amplitude   PhaseAngle'
            result = np.transpose(np.vstack((wavenumber, zeroAmplitude, amplitude, phaseAngle)))
        else:
            headLine = 'Wavenumber(cm-1)   Background   Amplitude'
            result = np.transpose(np.vstack((wavenumber, zeroAmplitude, amplitude)))
        np.savetxt(result_txt, result, fmt='%.3e', header=headLine)
        # Plot
        if kwargs['IF_PLOT_PHASE_ANGLE'] == True:
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
            axes[0].plot(self.x, amplitude)
            axes[1].plot(self.x, phaseAngle)
            axes[0].set_xlabel(r'Wavenumber (cm$^{-1}$)')
            axes[0].set_ylabel('Amplitude - KM')
            axes[1].set_xlabel(r'Wavenumber (cm$^{-1}$)')
            axes[1].set_ylabel('Phase')
            axes[1].set_xlim(X_RANGE[0], X_RANGE[1])
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
            axes.plot(self.x, amplitude)
            axes.set_xlabel(r'Wavenumber (cm$^{-1}$)')
            axes.set_ylabel('Amplitude - KM')
        plt.show()

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

        ##### Text output #####
        if self.ifIntegrate is True:
            integration = self.integration
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

        ##### Figures output #####
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
            fig, ax = plt.subplots()
            for i, singleSpecies in enumerate(np.transpose(integration)):
                if i == 0:
                    continue
                ax.plot(self.y, singleSpecies, label=self.intLabels[i - 1])

            # ax.set_xlim(0, 90)

            ax.set_yticks([])
            # ax.tick_params(axis='x', top='off', bottom='off')

            ax.legend(loc='best')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Integration Area (A. U.)')

            plt.show()


if __name__ == '__main__':
    testData = Series(**initParams)
    testData.generalPlot(**plotParams)
    testData.FourierTransformPlot(**FourierAnalysisParams)
    