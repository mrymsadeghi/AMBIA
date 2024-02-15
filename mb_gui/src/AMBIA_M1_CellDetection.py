import cv2 as cv
import numpy as np
from math import sqrt
from multiprocessing import  Pool
from skimage.feature import blob_log
import Switches_Static as st_switches
from Switches_Static import rx,cx
import time
class OUTPUTS:
    OUTPUT=[]
def calculate_norms(arr,X):
  return np.linalg.norm(X-arr.reshape(1,arr.shape[0]),axis=1)
def filter (arr):
    return np.any(arr<6)
def calculate_colocalized_blobs(blobs_log):
    blobs_log_=[]
    colocalized_blobs=[]
    colocalized_blobs_counts=[]
    for i in blobs_log:
        #filtering out all instances except the coordinates
        blobs_log_.append(np.array([x for x in i if isinstance(x, tuple)]))
    permutation=st_switches.coloc_permutation
    for i in permutation:
        no_coloc=False
        #setting two indexes for the first and second (last) channel
        index=st_switches.num_channels.index(i[0])
        index2=st_switches.num_channels.index(i[1])
        channel_to_start=blobs_log_[index]
        if len (channel_to_start)<1 or len(blobs_log_[index2])<1:
            #checking if the first channel has any cell detected, if not function set an empty list
            #and move to the next permute(if exist)
            no_coloc=True
            colocalized_blobs.append([])
            colocalized_blobs_counts.append(0)
            continue
        #calculating the distance between every pair of cells
        
        norms=np.apply_along_axis(calculate_norms,axis=1,arr=channel_to_start,X=blobs_log_[index2])
       
        coords=np.apply_along_axis(filter,axis=1,arr=norms)
        #print (coords)

        
        channel_to_start=channel_to_start[coords]
        if len(i)>2:
            for j in i[2:]:
                try :
                    index_=st_switches.num_channels.index(j)
                    norms=np.apply_along_axis(calculate_norms,axis=1,arr=channel_to_start,X=blobs_log_[index_])
                    coords=np.apply_along_axis(filter,axis=1,arr=norms)
                    channel_to_start=channel_to_start[coords]

                except:
                    no_coloc=True
                    break
        if no_coloc:
            colocalized_blobs.append([])
            colocalized_blobs_counts.append(0)
        else :    
            colocalized_blobs.append(list(channel_to_start))
            colocalized_blobs_counts.append(len(list(channel_to_start)))
    return colocalized_blobs_counts,colocalized_blobs




def MoG_detection(img_channel, maxsigma, numsigma, thresh, brain_mask_eroded):
    blobs_logs = []
    blobs_log = blob_log(img_channel, max_sigma=maxsigma, num_sigma=numsigma, threshold=thresh)
    for blob in blobs_log:
        y, x, r = blob
        if brain_mask_eroded[int(y), int(x)]==255:
            blobs_logs.append((int(y), int(x)))
    return blobs_log

    
def cfos_detection(args):

    """
    #args = [img_channel, minsigma, maxsigma, numsigma, thresh, brain_mask_eroded]
    #intermediate variable blobs_logs_all (numpy.ndarray)
    #Outputs blobs_log coords of detected cells in blevel image or patches of blevel image (in the case of parallel processing)
    #blobs_log: list of coords, (r, c)
    #!! blobs_log only includes list of blobs detected by this function and not the ones added manually by the user"""
  
    blobs_log = []
    blobs_logs_all = blob_log(args[0], min_sigma= args[1],max_sigma=args[2], num_sigma=args[3], threshold=args[4], overlap = st_switches.CELL_OVERLAP)
    brain_mask_eroded = args[5]
    for blob in blobs_logs_all:
        rb, cb, _ = blob
        if brain_mask_eroded[int(rb), int(cb)]==255:
            blobs_log.append((int(rb), int(cb)))

    return (blobs_log,args[-2],args[-1])
    



def log_result(result):
    OUTPUTS.OUTPUT.append(result)
def pool_cell_detection(img_channel, brain_mask_eroded, minsigma, maxsigma, numsigma, blobs_thresh):

    blobs_log = []
    OUTPUTS.OUTPUT.clear()
    r,c = img_channel.shape[0:2]
    #rx, cx = 12, 16

    rstep=r//rx
    cstep=c//cx
    patches=[]
    for i in range(rx):
        for j in range(cx):
            img_patch = img_channel[i*rstep:(i+1)*rstep,j*cstep:(j+1)*cstep]
            brain_mask_patch = brain_mask_eroded[i*rstep:(i+1)*rstep,j*cstep:(j+1)*cstep]
            patches.append([img_patch, minsigma, maxsigma, numsigma, blobs_thresh, brain_mask_patch,i,j])
    return patches,rx,cx,rstep,cstep
    

def calc_new_coords(patches,rx,cx,rstep,cstep):
    blobs_log=[]
    for patch in patches :
        patch,i,j=patch
        for point in patch:
            x1, y1= point
            x1+=i*rstep
            y1+=j*cstep
            blobs_log.append((int(x1), int(y1)))
    return blobs_log
