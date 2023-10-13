
import cv2 as cv
import numpy as np
from math import sqrt
from multiprocessing import Pool
from skimage.feature import blob_log




def rabies_detection(img_channel, blobs_thresh, minsize, brain_mask_eroded):
    """
    Uses morphological operations to detect blobs, Outputs blobs_log coords of detected cells in blevel image
    blobs_log: list of coords, (r, c)
    !! blobs_logs only includes list of blobs detected by this function and not the ones added manually by the user
    """
    blobs_log = []
    _, mask = cv.threshold(img_channel, blobs_thresh, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask_eroded = cv.erode(mask, kernel, iterations=1)
    components, _, stats, centers = cv.connectedComponentsWithStats(mask_eroded, 4, cv.CV_32S)

    for compnum in range(1, components):
        size = stats[compnum][-1]
        cb, rb = centers[compnum]
        if brain_mask_eroded[int(rb), int(cb)]==255:
            if size > minsize:
                blobs_log.append((int(rb), int(cb)))
    return blobs_log


def cfos_detection(args):
    """
    args = [img_channel, minsigma, maxsigma, numsigma, thresh, brain_mask_eroded]
    intermediate variable blobs_logs_all (numpy.ndarray)
    Outputs blobs_log coords of detected cells in blevel image or patches of blevel image (in the case of parallel processing)
    blobs_log: list of coords, (r, c)
    !! blobs_log only includes list of blobs detected by this function and not the ones added manually by the user
    """
    blobs_log = []
    img_channel_adjusted = cv.convertScaleAbs(args[0], alpha=2, beta=0) # beta > Brightness control (0-100)
    blobs_logs_all = blob_log(img_channel_adjusted, min_sigma= args[1],max_sigma=args[2], num_sigma=args[3], threshold=args[4])
    brain_mask_eroded = args[5]
    for blob in blobs_logs_all:
        rb, cb, _ = blob
        if brain_mask_eroded[int(rb), int(cb)]==255:
            blobs_log.append((int(rb), int(cb)))
    return blobs_log


def MoG_detection(img_channel, maxsigma, numsigma, thresh, brain_mask_eroded):
    """
    Another  cell detection function that can be added by the user
    """
    blobs_logs = []
    blobs_log = blob_log(img_channel, max_sigma=maxsigma, num_sigma=numsigma, threshold=thresh)
    for blob in blobs_log:
        y, x, r = blob
        if brain_mask_eroded[int(y), int(x)]==255:
            blobs_logs.append((int(y), int(x)))
    return blobs_log

def calculate_colocalized_blobs(blobs_log_r, blobs_log_g):
    """
    Finds the collocalized cells between different channels, e.g. red and green channel. 
    A cell is considered a colocalized if their center are less than a specified distance appart here specified as DIST
    """
    DIST = 6
    colocalized_blobs = []
    for blob in blobs_log_g:
        rb, cb = blob  # blob locations in level 2
        # All the locations in the patch level
        for blobr in blobs_log_r:
            rbr, cbr = blobr
            dist = int(sqrt((rbr - rb) ** 2 + (cbr - cb) ** 2))
            if dist < DIST:
                colocalized_blobs.append((rbr, cbr))
    matchcount = len(colocalized_blobs)
    return matchcount, colocalized_blobs


def pool_cell_detetcion(img_channel, brain_mask_eroded, minsigma, maxsigma, numsigma, blobs_thresh):
    blobs_log = []
    r,c = img_channel.shape[0:2]
    rstep=r//3
    cstep=c//4
    patches=[]
    for i in range(3):
        for j in range(4):
            img_patch = img_channel[i*rstep:(i+1)*rstep,j*cstep:(j+1)*cstep]
            brain_mask_patch = brain_mask_eroded[i*rstep:(i+1)*rstep,j*cstep:(j+1)*cstep]
            patches.append([img_patch, minsigma, maxsigma, numsigma, blobs_thresh, brain_mask_patch])
    with Pool(4) as p:
        allPoints=p.map(cfos_detection, patches)
    count=0
    pcount=0
    for i in range(3):
        for j in range(4):
            for point in allPoints[count]:
                x1, y1= point
                x1+=i*rstep
                y1+=j*cstep
                blobs_log.append((int(x1), int(y1)))
                pcount+=1
            count+=1
    return blobs_log