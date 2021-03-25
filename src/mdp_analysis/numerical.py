import numpy as np


### CHeck if significant ###
def check_if_significant_np(data_in, threshold):
    """
    Check if significant with regard to threshold.
    Args:
        data_in (float array): input data.
        threshold (float): the threshold value.
    Returns:
        data_out (float array): significant columns.
        indicies (integer array): indicies of significant values.
    """
    indices = np.nonzero(np.var(efield, axis=1) > threshold)
    data_out = data_in[indices]
    return data_out, indices



### Do Fourier transformation ###
def do_DFT(data_in, tmax):
    """
    Do Discrete Fourier transformation.
    Args:
        data_in (array): input data.
        tmax (float): sample frequency.
    Returns:
        data_s (complex array): output array.
        data_w (array): the Discrete Fourier Transform sample frequencies.
    """
    data_s = np.fft.rfft(data_in)
    data_w = np.fft.rfftfreq(tmax)
    return data_s, data_w

### Do Fourier transformation ###
def do_positive_DFT(data_in, tmax):
    """
    Do Discrete Fourier transformation and take POSITIVE frequency component part.
    Args:
        data_in (array): input data.
        tmax (float): sample frequency.
    Returns:
        data_s (array): output array with POSITIVE frequency component part.
        data_w (array): the Discrete Fourier Transform sample frequencies POSITIVE frequency component part.
    """
    data_s = np.fft.fft(data_in)
    data_w = np.fft.fftfreq(tmax)
    # only take the positive frequency components
    return data_w[0:tmax//2], data_s[0:tmax//2]



### Claculate autocorelation ###
def calc_auto(wavef):
    """
    Claculate autocorelation.
    Args:
        wavef (complex array): input array.
    Returns:
        autocofu (array): output array.
    """
    aucofu = np.zeros(len(wavef[0]),dtype = complex)
    for i in range(0,len(wavef[0])):
        aucofu[i] = np.sum(wavef[:,0]*np.conjugate(wavef[:,i]))
    return aucofu
