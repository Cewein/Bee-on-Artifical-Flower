import cv2 as cv
import skvideo.io

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


##### Video processing #####
# open a video
def openFirstFrame(src: str) -> np.ndarray:
    return skvideo.io.vread(src, num_frames=1)[0]



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

##### Regions of interest - display function #####

def drawBoundingBox(rois: np.ndarray, img: np.ndarray) -> None:

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    for i in range(len(rois)):
        # Create a Rectangle patch
        rect = patches.Rectangle((rois[i][0], rois[i][1]), rois[i][2], rois[i][3], linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
    return