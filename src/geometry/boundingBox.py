from matplotlib import patches, pyplot as plt
import numpy as np

def getBBoxCenter(bboxes: np.ndarray)-> np.ndarray:
    if bboxes.shape[1] != 4: raise Exception("bboxes must be with dim [n,4], n being the number of bbox")

    #bbox is in mode XYWH
    if bboxes[0][0] >= bboxes[0][2]:
        return bboxes[:,:2] + (bboxes[:,2:] // 2)
    #bbox is in mode minmaxXY
    else:
        return bboxes[:,:2] + ((bboxes[:,2:] - bboxes[:,2:]) // 2)
    
def getAllPoints(bboxes: np.ndarray)-> np.ndarray:
    if bboxes.shape[1] != 4: raise Exception("bboxes must be with dim [n,4], n being the number of bbox")

    #bbox is in mode XYWH
    if bboxes[0][0] >= bboxes[0][2]:
        bboxes = XYWHtominmaxXY(bboxes)

    minXY = bboxes[:,:2]
    maxXY = bboxes[:,2:]

    allPoint = np.array([minXY[:,0],minXY[:,1]])

    allPoint =  np.hstack((allPoint, np.array([maxXY[:,0],minXY[:,1]])))
    allPoint =  np.hstack((allPoint, np.array([maxXY[:,0],maxXY[:,1]])))
    allPoint =  np.hstack((allPoint, np.array([minXY[:,0],maxXY[:,1]])))
    
    return allPoint
    
#need to be in minmaxXY
def getGridBBox(roiArray: np.ndarray) -> np.ndarray:
    minXY = np.min(roiArray[:,:2], axis=0)
    maxXY = np.max(roiArray[:,2:], axis=0)
    cornersBbox = [[minXY[0],minXY[1]],
               [maxXY[0],minXY[1]],
               [maxXY[0],maxXY[1]],
               [minXY[0],maxXY[1]]]
    
    return np.array(cornersBbox)


## change the definition of bouding box
def XYWHtominmaxXY(BBoxArray: np.ndarray) -> np.ndarray:
    if BBoxArray.shape[1] != 4:
        raise Exception('BBox array must be with dim [n,4], n being the number of BBox')
    
    #replace WH to min XY + WH = max XY
    BBoxArray[:,2:] = BBoxArray[:,:2] + BBoxArray[:,2:]
    
    return BBoxArray

def minmaxXYtoXYWH(BBoxArray: np.ndarray) -> np.ndarray:
    if BBoxArray.shape[1] != 4:
        raise Exception('BBox array must be with dim [n,4], n being the number of BBox')
    
    #replace max XY to max XY - min XY = WH
    BBoxArray[:,2:] =  BBoxArray[:,2:] - BBoxArray[:,:2]
    
    return BBoxArray

#change normal space to XYWH inage space
def normSpaceToImgSpace(BBoxArray: np.ndarray, img: np.ndarray):

    resolutionY,resolutionX,_ = img.shape

    w = np.int32(BBoxArray[:,2] * resolutionX)
    h = np.int32(BBoxArray[:,3] * resolutionY)

    x = np.int32(BBoxArray[:,0] * resolutionX - w/2)
    y = np.int32(BBoxArray[:,1] * resolutionY - h/2)

    return np.array([x,y,w,h]).T

##### display function #####

def drawWithCategory(img: np.ndarray, BBoxs: np.ndarray, categories: list) -> None:

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 15))

    # Display the image
    ax.imshow(img)

    for i in range(len(BBoxs)):
        
        id = np.int32(BBoxs[i][0])
        colors = ['b','orange','w']

        w = BBoxs[i][3]
        h = BBoxs[i][4]

        x = BBoxs[i][1]
        y = BBoxs[i][2]

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