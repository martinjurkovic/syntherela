import numpy as np

def get_x_tick_width_coef(N):
            if N == 5:
                return 2
            elif N == 4:
                return 1.5
            elif N == 3:
                return 1
            elif N == 2:
                return 0.5
            else:
                return 0
            
def get_bins(data):
    if data.dtype.name == 'category' or data.dtype.name == 'object':
        return len(data.unique())
    if data.dtype.name == 'bool':
        return 2
    if data.dtype.name == 'datetime64' or data.dtype.name == 'datetime64[ns]':
        return 'auto'
    return np.histogram_bin_edges(data.dropna())