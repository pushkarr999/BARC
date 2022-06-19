from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
import numpy as np


def peak(data, time, thres):
    # Find peaks
    peaks = find_peaks(data, height=thres, distance=15)  # thres,dist
    height = peaks[1]['peak_heights']  # list containing the height of the peaks
    peak_pos = time[peaks[0]]  # list containing the positions of the peaks
    peak_pos = peak_pos.to_numpy(dtype='float32')
    return height, peak_pos


def cwt(data, time, thres):
    peakind = find_peaks_cwt(data, np.arange(15, 50))
    peakind = peakind[data[peakind] > thres]
    height = data[peakind]
    peak_pos = time[peakind]
    height = height.to_numpy(dtype='float32')
    peak_pos = peak_pos.to_numpy(dtype='float32')
    return height, peak_pos


def peakdet(data, time, thres, x=None):
    maxtab = []
    mintab = []
    height = []
    distance = 15

    if x is None:
        x = np.arange(len(data))

    data = np.asarray(data)
    delta = 10
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(data) - distance):
        this = data[i + distance]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                if (mx > thres):
                    maxtab.append(mxpos)
                    height.append(mx)
                    mn = this
                    mnpos = x[i]
                    lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    maxtab = time[maxtab]
    height = np.array(height, dtype='float32')
    maxtab = np.array(maxtab, dtype='float32')
    # print(height,maxtab)
    return height, maxtab


def detect_peaks(signal, time, threshold):
    """By Tony Betramelli -  Performs peak detection on three steps: root mean square, peak to
	average ratios and first order logic.
	threshold used to discard peaks too small """
    # compute root mean square
    thres = 0.5
    root_mean_square = np.sqrt(np.sum(np.square(signal) / len(signal)))
    # compute peak to average ratios
    ratios = np.array([pow(x / root_mean_square, 2) for x in signal])
    # apply first order logic
    peaks = (ratios > np.roll(ratios, 1)) & (ratios > np.roll(ratios, -1)) & (ratios > thres)
    # optional: return peak indices
    peak_indexes = []
    for i in range(0, len(peaks)):
        if peaks[i] and signal[i]>threshold:
            #print(peaks[i])
            peak_indexes.append(i)
    height = signal[peak_indexes]
    times = time[peak_indexes]
    height = np.array(height, dtype='float32')
    times = np.array(times, dtype='float32')
    #print(peak_indexes)
    return height, times