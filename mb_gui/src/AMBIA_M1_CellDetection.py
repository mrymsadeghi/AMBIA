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

def calculate_colocalized_blobs(blobs_log):#blobs_log_r, blobs_log_g):
    blobs_log_=[]
    colocalized_blobs=[]
    colocalized_blobs_counts=[]
    for i in blobs_log:
        blobs_log_.append(np.array([x for x in i if isinstance(x, tuple)]))
    permutation=st_switches.coloc_permutation
    for i in permutation:
        no_coloc=False
        index=st_switches.num_channels.index(i[0])
        index2=st_switches.num_channels.index(i[1])
        channel_to_start=blobs_log_[index]
        if len (channel_to_start)<1 or len(blobs_log_[index2])<1:
            no_coloc=True
            colocalized_blobs.append([])
            colocalized_blobs_counts.append(0)
            continue
        #print ("COLOC",channel_to_start)
        norms=np.apply_along_axis(calculate_norms,axis=1,arr=channel_to_start,X=blobs_log_[index2])

        coords=np.where(norms<6)

        try : channel_to_start=channel_to_start[np.unique(coords[1])]#[np.all(norms<6,axis=0)]
        except :channel_to_start=channel_to_start[np.unique(coords[0])]
        if len(i)>2:
            for j in i[2:]:
                try :
                    index_=st_switches.num_channels.index(j)
                    norms=np.apply_along_axis(calculate_norms,axis=1,arr=channel_to_start,X=blobs_log_[index_])
                    coords=np.where(norms<6)
                    try : channel_to_start=channel_to_start[np.unique(coords[1])]
                    except : channel_to_start=channel_to_start[np.unique(coords[0])]
                except:
                    no_coloc=True
                    break
        if no_coloc:
            colocalized_blobs.append([])
            colocalized_blobs_counts.append(0)
        else :    
            colocalized_blobs.append(list(channel_to_start))
            colocalized_blobs_counts.append(len(list(channel_to_start)))
    return colocalized_blobs_counts,colocalized_blobs#len (colocalized_blobs), colocalized_blobs
    #colocalized_blobs = []
    #result = np.apply_along_axis(find_coloc,arr=)



def MoG_detection(img_channel, maxsigma, numsigma, thresh, brain_mask_eroded):
    blobs_logs = []
    blobs_log = blob_log(img_channel, max_sigma=maxsigma, num_sigma=numsigma, threshold=thresh)
    for blob in blobs_log:
        y, x, r = blob
        if brain_mask_eroded[int(y), int(x)]==255:
            blobs_logs.append((int(y), int(x)))
    return blobs_log

    
def cfos_detection(args):
    print (f"patch {args[-2],args[-1]} started.")
    z=time.time()
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
    print ("1 itter took ",time.time()-z,args[-2],args[-1])
    """file=open(f"test{args[-2],args[-1]}.txt","w")
    file.write((blobs_log,args[-2],args[-1]))"""
    return (blobs_log,args[-2],args[-1])
    #return blobs_log
    

def double_cfos_detection(args):
    import time
    timer=time.time()
    """args = [img_channel, minsigma, maxsigma, numsigma, thresh, brain_mask_eroded]
    intermediate variable blobs_logs_all (numpy.ndarray)
    Outputs blobs_log coords of detected cells in blevel image or patches of blevel image (in the case of parallel processing)
    blobs_log: list of coords, (r, c)
    !! blobs_log only includes list of blobs detected by this function and not the ones added manually by the user
    """

    blobs_log_r = []
    blobs_logs_all = blob_log(args[0], min_sigma= args[1], max_sigma=args[2], num_sigma=args[3], threshold=args[4], overlap = st_switches.CELL_OVERLAP)
    brain_mask_eroded = args[10]
    for blob in blobs_logs_all:
        rb, cb, _ = blob
        if brain_mask_eroded[int(rb), int(cb)]==255:
            blobs_log_r.append((int(rb), int(cb)))
    blobs_log_g = []
    #blobs_logs_all = blob_log(args[5], min_sigma= args[1], max_sigma=args[6], num_sigma=args[7], threshold=args[8], overlap = st_switches.CELL_OVERLAP)
    blobs_logs_all = blob_log(args[5], min_sigma= args[6], max_sigma=args[7], num_sigma=args[8], threshold=args[9], overlap = st_switches.CELL_OVERLAP)
    
    for blob in blobs_logs_all:
        rb, cb, _ = blob

        if brain_mask_eroded[int(rb), int(cb)]==255:
            blobs_log_g.append((int(rb), int(cb)))
    print (time.time()-timer," in double detection after two blob_log")

 
    matchcount, colocalized_blobs = calculate_colocalized_blobs(blobs_log_r, blobs_log_g)
    print (time.time()-timer," after  coloc calc",matchcount)
    return blobs_log_r, blobs_log_g, matchcount, colocalized_blobs


def log_result(result):
    OUTPUTS.OUTPUT.append(result)
def pool_cell_detection(img_channel, brain_mask_eroded, minsigma, maxsigma, numsigma, blobs_thresh, cellcolor):
    print(f"Running {cellcolor}")
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
    print ("Running cFos on",len(patches),"Patches")
    """pool = Pool(processes=20)
    pool2=Pool(processes=20)
    pool3=Pool(processes=20)
    pool4=Pool(processes=20)
    slicer=len(patches)
    processes=[]
    for i in patches:
        processes.append(pool.apply_async(cfos_detection,[i]))
    print ("collecting results")
    for i in processes:
        OUTPUTS.OUTPUT.append(i.get())
        #OUTPUTS.OUTPUT.append(result.get())
    pool.close()
    pool.join()"""
    with Pool(10) as p:
        allPoints=p.map(cfos_detection, patches[:192])
        
    p.join()
    print (len (OUTPUTS.OUTPUT),"OUTPUT")
    for i in range(rx):
        for j in range(cx):
            for point in allPoints[count]:
                x1, y1= point
                x1+=i*rstep
                y1+=j*cstep
                blobs_log.append((int(x1), int(y1)))
                pcount+=1
            count+=1
    """for points,i,j in OUTPUTS.OUTPUT:
        for point in points:
            x1, y1= point
            x1+=i*rstep
            y1+=j*cstep
            blobs_log.append((int(x1), int(y1)))"""
        #pcount+=1

    return blobs_log

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
def double_pool_cell_detection(img_channel_r, img_channel_g, brain_mask_eroded, minsigma_r, minsigma_g, maxsigma_r, numsigma_r, red_blobs_thresh, maxsigma_g, numsigma_g, green_blobs_thresh):
    import time
    timer1=time.time()
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
            patches.append([img_patch_r, minsigma_r, maxsigma_r, numsigma_r, red_blobs_thresh,
                             img_patch_g, minsigma_g, maxsigma_g, numsigma_g, green_blobs_thresh,brain_mask_patch])
    print (time.time()-timer1 ,"timer 3")
    print (len (patches),"patchesssssssssssssssssssssss")
    with Pool(10) as p:
        all_points_bags = p.map(double_cfos_detection, patches)
        #blobs_log_r0, blobs_log_g0, matchcount, colocalized_blobs0 = p.map(double_cfos_detection, patches)
    print ("timer final ",time.time()-timer1)
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
    