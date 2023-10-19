import cv2 as cv
import numpy as np
from math import sqrt
from multiprocessing import Pool
from skimage.feature import blob_log
import Switches_Static as st_switches

def calculate_colocalized_blobs(blobs_log_r, blobs_log_g):

    colocalized_blobs = []
    for blob in blobs_log_g:
        rb, cb = blob  # blob locations in level 2
        # All the locations in the patch level
        for blobr in blobs_log_r:
            rbr, cbr = blobr
            dist = int(sqrt((rbr - rb) ** 2 + (cbr - cb) ** 2))
            if dist < 6:
                colocalized_blobs.append((rbr, cbr))
                continue
    matchcount = len(colocalized_blobs)
    return matchcount, colocalized_blobs


def MoG_detection(img_channel, maxsigma, numsigma, thresh, brain_mask_eroded):
    blobs_logs = []
    blobs_log = blob_log(img_channel, max_sigma=maxsigma, num_sigma=numsigma, threshold=thresh)
    for blob in blobs_log:
        y, x, r = blob
        if brain_mask_eroded[int(y), int(x)]==255:
            blobs_logs.append((int(y), int(x)))
    return blobs_log

def double_cfos_detection(args):
    
    """args = [img_channel, minsigma, maxsigma, numsigma, thresh, brain_mask_eroded]
    intermediate variable blobs_logs_all (numpy.ndarray)
    Outputs blobs_log coords of detected cells in blevel image or patches of blevel image (in the case of parallel processing)
    blobs_log: list of coords, (r, c)
    !! blobs_log only includes list of blobs detected by this function and not the ones added manually by the user
    """

    blobs_log_r = []
    img_channel_adjusted_r = cv.convertScaleAbs(args[0], alpha=2, beta=0) # beta > Brightness control (0-100)
    blobs_logs_all = blob_log(img_channel_adjusted_r, min_sigma= args[1], max_sigma=args[2], num_sigma=args[3], threshold=args[4], overlap = st_switches.CELL_OVERLAP)
    brain_mask_eroded = args[9]
    for blob in blobs_logs_all:
        rb, cb, _ = blob
        if brain_mask_eroded[int(rb), int(cb)]==255:
            blobs_log_r.append((int(rb), int(cb)))
    blobs_log_g = []
    img_channel_adjusted_g = cv.convertScaleAbs(args[5], alpha=2, beta=0) # beta > Brightness control (0-100)
    blobs_logs_all = blob_log(img_channel_adjusted_g, min_sigma= args[1], max_sigma=args[6], num_sigma=args[7], threshold=args[8], overlap = st_switches.CELL_OVERLAP)
    for blob in blobs_logs_all:
        rb, cb, _ = blob
        if brain_mask_eroded[int(rb), int(cb)]==255:
            blobs_log_g.append((int(rb), int(cb)))

    matchcount, colocalized_blobs = calculate_colocalized_blobs(blobs_log_r, blobs_log_g)
    return blobs_log_r, blobs_log_g, matchcount, colocalized_blobs






def pool_cell_detection(img_channel, brain_mask_eroded, minsigma, maxsigma, numsigma, blobs_thresh, cellcolor):
    print(f"Running {cellcolor}")
    blobs_log = []
    r,c = img_channel.shape[0:2]
    rx, cx = 12, 16

    rstep=r//rx
    cstep=c//cx
    patches=[]
    for i in range(rx):
        for j in range(cx):
            img_patch = img_channel[i*rstep:(i+1)*rstep,j*cstep:(j+1)*cstep]
            brain_mask_patch = brain_mask_eroded[i*rstep:(i+1)*rstep,j*cstep:(j+1)*cstep]
            patches.append([img_patch, minsigma, maxsigma, numsigma, blobs_thresh, brain_mask_patch])
    with Pool(10) as p:
        allPoints=p.map(cfos_detection, patches)
    count=0
    pcount=0
    for i in range(rx):
        for j in range(cx):
            for point in allPoints[count]:
                x1, y1= point
                x1+=i*rstep
                y1+=j*cstep
                blobs_log.append((int(x1), int(y1)))
                pcount+=1
            count+=1
    return blobs_log

def double_pool_cell_detection(img_channel_r, img_channel_g, brain_mask_eroded, minsigma, maxsigma_r, numsigma_r, red_blobs_thresh, maxsigma_g, numsigma_g, green_blobs_thresh):

    r,c = img_channel_r.shape[0:2]
    rx, cx = 12, 16
    rstep=r//rx
    cstep=c//cx
    patches=[]
    for i in range(rx):
        for j in range(cx):
            img_patch_r = img_channel_r[i*rstep:(i+1)*rstep,j*cstep:(j+1)*cstep]
            img_patch_g = img_channel_g[i*rstep:(i+1)*rstep,j*cstep:(j+1)*cstep]
            brain_mask_patch = brain_mask_eroded[i*rstep:(i+1)*rstep,j*cstep:(j+1)*cstep]
            patches.append([img_patch_r, minsigma, maxsigma_r, numsigma_r, red_blobs_thresh, img_patch_g, maxsigma_g, numsigma_g, green_blobs_thresh,brain_mask_patch])
    with Pool(10) as p:
        all_points_bags = p.map(double_cfos_detection, patches)
        #blobs_log_r0, blobs_log_g0, matchcount, colocalized_blobs0 = p.map(double_cfos_detection, patches)
    blobs_log_r = []
    blobs_log_g = []
    colocalized_blobs = []
    count=0
    pcount=0    
    matchcount = 0
    for i in range(rx):
        for j in range(cx):
            blobs_log_r0, blobs_log_g0, matchcount0, colocalized_blobs0 = all_points_bags[count]
            for point in blobs_log_r0:
                x1, y1= point
                x1+=i*rstep
                y1+=j*cstep
                blobs_log_r.append((int(x1), int(y1)))

            for point in blobs_log_g0:
                x1, y1= point
                x1+=i*rstep
                y1+=j*cstep
                blobs_log_g.append((int(x1), int(y1)))

            for point in colocalized_blobs0:
                x1, y1= point
                x1+=i*rstep
                y1+=j*cstep
                colocalized_blobs.append((int(x1), int(y1)))
            matchcount += matchcount0       
            count+=1
    return blobs_log_r, blobs_log_g, matchcount, colocalized_blobs  
    
    
def cfos_detection(args):
    """
    #args = [img_channel, minsigma, maxsigma, numsigma, thresh, brain_mask_eroded]
    #intermediate variable blobs_logs_all (numpy.ndarray)
    #Outputs blobs_log coords of detected cells in blevel image or patches of blevel image (in the case of parallel processing)
    #blobs_log: list of coords, (r, c)
    #!! blobs_log only includes list of blobs detected by this function and not the ones added manually by the user"""
  
    blobs_log = []
    img_channel_adjusted = cv.convertScaleAbs(args[0], alpha=2, beta=0) # beta > Brightness control (0-100)
    blobs_logs_all = blob_log(img_channel_adjusted, min_sigma= args[1],max_sigma=args[2], num_sigma=args[3], threshold=args[4], overlap = st_switches.CELL_OVERLAP)
    brain_mask_eroded = args[5]
    for blob in blobs_logs_all:
        rb, cb, _ = blob
        if brain_mask_eroded[int(rb), int(cb)]==255:
            blobs_log.append((int(rb), int(cb)))
    return blobs_log
    