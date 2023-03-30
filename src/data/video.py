import numpy as np
import imageio.v3 as iio

##### Video processing #####

# open a video
def openFrame(src: str, frame_num: int) -> np.ndarray:
    return iio.imread(src,index=frame_num,plugin="pyav")