import matplotlib.pyplot as plt
import numpy as np
import skimage.filters

def thresholdingRGB(img: np.ndarray) -> np.ndarray:
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    rTresh = skimage.filters.threshold_minimum(r)
    gTresh = skimage.filters.threshold_minimum(g)
    bTresh = skimage.filters.threshold_minimum(b)

    r[r < rTresh] = 0
    g[g < rTresh] = 0
    b[b < rTresh] = 0

    return np.dstack((r,g,b))