from matplotlib import pyplot as plt
import numpy as np
import imageio.v3 as iio
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage import transform
from skimage.color import rgb2gray
from skimage.filters import gaussian

##### Video processing #####

# open a video
def openFrame(src: str, frameNumber: int) -> np.ndarray:
    return iio.imread(src,index=frameNumber,plugin="pyav")