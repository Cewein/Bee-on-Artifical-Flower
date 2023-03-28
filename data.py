import cv2 as cv
import ffmpeg

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


##### Video processing #####
# open a video
def openFrame(src: str, frame_num: int) -> np.ndarray:
    probe = ffmpeg.probe(src)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])

    out, err = (
        ffmpeg
        .input(src)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, height, width, 3])
    )

    return video[frame_num,:,:,:]


##### Regions of interest #####

def getPointFromImage(image: np.ndarray,windowName: str) -> np.ndarray:
    img = image.copy()
    pointArray = []

    def click_event(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            pointArray.append([x,y])
            
            # draw point on the image
            cv.circle(img, (x,y), 2, (0,255,255), -1)

    cv.namedWindow(windowName)
    cv.setMouseCallback(windowName, click_event)

    while True:
        cv.imshow(windowName,img)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        
    cv.destroyAllWindows()

    return np.array(pointArray)


def getRegionsOfInterest(image: np.ndarray, windowName: str) -> np.ndarray:

    cv.namedWindow(windowName)
    pointArray = cv.selectROIs(windowName, cv.cvtColor(image, cv.COLOR_RGB2BGR))
    cv.destroyAllWindows()

    return np.array(pointArray)

##### processing of the ROI (regions of interest) #####

# generate offset for bounding boxes
def generateOffset(roi: list, imageShape: tuple, derivation: float) -> tuple:
    extX = np.int_(roi[2]*derivation)

    extXBot = np.clip(np.random.randint(0, extX), None, roi[0])
    extXTop = np.clip(np.random.randint(0, extX), None, imageShape[1]-(roi[0]+roi[2]))

    extY = np.int_(roi[3]*derivation)

    extYBot = np.clip(np.random.randint(0, extY), None, roi[1])
    extYTop = np.clip(np.random.randint(0, extY), None, imageShape[0]-(roi[1]+roi[3]))

    return (extXBot,extXTop,extYBot,extYTop)

def extractAndSave(rois:np.ndarray, img:np.ndarray, name: str, path: str, derivation: float = 0.3):

    dictArr = []
    
    for i in range(0,len(rois)):
        roi = rois[i]

        extXBot, extXTop, extYBot, extYTop = generateOffset(roi, img.shape, derivation)

        xb = roi[0]-extXBot
        xt = roi[0]+roi[2]+extXTop

        yb = roi[1]-extYBot
        yt = roi[1]+roi[3]+extYTop

        filename = f"flower-{name}-{i}.png"

        #slicing is row major
        subPic = img[yb:yt,xb:xt]

        #bounding box is collum major
        subDict = {"name":filename, "roi": [extXBot,
                                            extYBot,
                                            roi[2],
                                            roi[3]]}
        dictArr.append(subDict)

        plt.imsave(path + filename,subPic)
    
    np.save(path+f"/dict-{name}", np.array(dictArr))

def XYWHtominmaxXY(roiArray: np.ndarray) -> np.ndarray:
    if roiArray.shape[1] != 4:
        raise Exception('Roi array must be with dim [n,4], n being the number of roi')
    
    #replace WH to min XY + WH = max XY
    roiArray[:,2:] = roiArray[:,:2] + roiArray[:,2:]
    
    return roiArray

def minmaxXYtoXYWH(roiArray: np.ndarray) -> np.ndarray:
    if roiArray.shape[1] != 4:
        raise Exception('Roi array must be with dim [n,4], n being the number of roi')
    
    #replace max XY to max XY - min XY = WH
    roiArray[:,2:] =  roiArray[:,2:] - roiArray[:,:2]
    
    return roiArray


##### Regions of interest - display function #####

def drawBoundingBox(img: np.ndarray, rois: np.ndarray, categories: list) -> None:

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 15))

    # Display the image
    ax.imshow(img)

    for i in range(len(rois)):
        
        id = rois[i][1]
        colors = ['g','w','orange','b']

        x,y,w,h = rois[i][2], rois[i][3], rois[i][4], rois[i][5]

        # Create a Rectangle patch
        rect = patches.Rectangle((x,y),w,h, linewidth=1, edgecolor=colors[id], facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            categories[id],
            bbox={"facecolor": colors[id], "alpha": 0.4},
            clip_box=ax.clipbox, # type: ignore
            clip_on=True,
            fontsize='xx-small'
        )

    plt.show()