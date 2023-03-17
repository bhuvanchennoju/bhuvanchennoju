
import numpy as np
from scipy.signal import butter, lfilter

class ButterFilter:
    # add the typing and the docstring to the class

    """ Butterworth filter class for filtering the data with the given order, sampling frequency, lowcut and highcut values. 
    The filter type can be band, low or high. The default filter type is band.   
    
    Parameters
    ----------
    order : int --Order of the filter
    fs : float
        Sampling frequency
    lowcut : float
        Lowcut value
    highcut : float
        Highcut value
    btype : str
        Filter type. Default is band. Can be band, low or high.
            
    Methods 
    -------
    filter(data)
        Filters the data with the given order, sampling frequency, lowcut and highcut values.

    Examples
    --------
    >>> from butter_filter import ButterFilter
    >>> butter_filter = ButterFilter(3, 13.3333, 0.5, 1, 'band')
    >>> butter_filter.filter(data)

    """

    def __init__(self, order : int, fs :float, lowcut :float, highcut :float, btype :str = 'band') -> None:
        self.order = order
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.btype = btype
        self.nyq = 0.5 * self.fs  #Nyquist frequency is half of the sampling frequency
        self.low = self.lowcut / self.nyq
        self.high = self.highcut / self.nyq

        if self.btype == 'band':
            self.b, self.a = butter(self.order, [self.low, self.high], btype='band')
        elif self.btype == 'low':
            self.b, self.a = butter(self.order, self.low, btype='low')
        elif self.btype == 'high':
            self.b, self.a = butter(self.order, self.high, btype='high')
        else:
            raise ValueError('btype must be band, low or high')

        self.b, self.a = butter(self.order, self.low, btype='low')

    
    def apply(self, data : np.ndarray) -> np.ndarray:
        y = lfilter(self.b, self.a, data)
        return y