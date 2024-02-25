import numpy as np
from pathlib import Path
import Switches_Static as st_switches
from skimage import measure

import cv2 as cv
def check_switch(parameter):
    global tf
    if parameter.section_classifier_on or parameter.segmentation_1_20_on or parameter.section_QL_on:
        import tensorflow
        tf=tensorflow
def rotate_by_angle(image, angle, center = None, scale = 1.0):
        (h, w) = image.shape[:2]
        borderValue=(0,0,0)
        pad_value=(0, 0, 0)
        if st_switches.Bright_field:
            pad_value=(255, 255, 255)
            
            borderValue=(255,255,255)
        padded_image = cv.copyMakeBorder(image, st_switches.rotation_padding, st_switches.rotation_padding,
                                              st_switches.rotation_padding, st_switches.rotation_padding, 
                                              cv.BORDER_CONSTANT,pad_value)
        h,w=h+(st_switches.rotation_padding*2),w+(st_switches.rotation_padding*2)
        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        
        M = cv.getRotationMatrix2D(center, angle, scale)
        rotated = cv.warpAffine(padded_image, M, (w, h) ,borderValue=borderValue)

        return rotated

def rotate_img(img):
    img_rot = np.zeros((img.shape[1], img.shape[0], img.shape[2]))      ###########  optimize
    img_rot[:, :, 0] = np.rot90(img[:, :, 0], k=-1, axes=(0, 1))
    img_rot[:, :, 1] = np.rot90(img[:, :, 1], k=-1, axes=(0, 1))
    img_rot[:, :, 2] = np.rot90(img[:, :, 2], k=-1, axes=(0, 1))
    return img_rot


def equalize_img(image):                                                 ###########  optimize
    b, g, r = cv.split(image)
    diff = int(18 / (np.mean(g) + 1))
    g_lighter = g
    if diff != 0:
        g_lighter = g * diff      
    image_equalized = cv.merge((b, g_lighter, r))
    return image_equalized


def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.
    Args:
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. `order=2` for L2 norm).
    Returns:
        A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)


def standardize(x):
    x = np.array(x, dtype='float64')
    x -= np.min(x)
    x /= np.percentile(x, 98)
    x[x > 1] = 1
    return x

def preprocessing(img):
    image = np.array(img)   
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image = np.zeros_like(image)
    image[:,:,0] = gray
    image[:,:,1] = gray
    image[:,:,2] = gray
    image = standardize(image)
    return image

def load_images(path, size_y, size_x):
    images = []
    EXTENSIONS = {'.png', '.jpg'}
    for img_path in Path(path).glob('*'):
        if img_path.suffix in EXTENSIONS:
            img = cv.imread(str(img_path))
            img = cv.resize(img, (size_y, size_x))
            img = preprocessing(img)

            images.append(img)
        
    return np.array(images)

def load_image(path, size_y, size_x):
    
    img = cv.imread(str(path))
    img = cv.resize(img, (size_y, size_x))
    img = preprocessing(img)

    return img

def get_mask(img, model):
    pred_mask = model.predict(img[tf.newaxis, ...])
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
                  
    return pred_mask[0].numpy()

def threshold(p, v):
    if p == v:
        return 1
    return 0

def remove_outliers(mask):
    labels_mask = measure.label(mask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    
    return labels_mask

def get_regions(mask, size_x=256, size_y=256):

    present_region = [False, False, False]
    total_num_of_pixels = size_x*size_y #256*256
    min_size= total_num_of_pixels*0.10
    
    le = []
    for row in mask:
        new_row = [threshold(p, 1) for p in row]
        le += [new_row]

    le = np.array(le)
    le = remove_outliers(le)
    
    if np.sum(le) > min_size:
        present_region[0] = True
    
    re = []
    for row in mask:
        new_row = [threshold(p, 2) for p in row]
        re += [new_row]

    re = np.array(re)
    re = remove_outliers(re)
    
    if np.sum(re) > min_size:
        present_region[1] = True
    
    bs = []
    for row in mask:
        new_row = [threshold(p, 3) for p in row]
        bs += [new_row]

    bs = np.array(bs)
    bs = remove_outliers(bs)
    
    if np.sum(bs) > min_size:
        present_region[2] = True

    return [le, re, bs], present_region

def get_crop(img, min_h, max_h, min_w, max_w, size_x=256, size_y=256):

    img = np.array(img)
    
    dim = len(img)
    real_min_w = int(min_w*dim/size_x)
    real_max_w = int(max_w*dim/size_x)
    real_min_h = int(min_h*dim/size_y)
    real_max_h = int(max_h*dim/size_y)
    
    return img[real_min_h:real_max_h, real_min_w:real_max_w]

def get_box(mask):
    idx = np.where(mask==1)
    min_h = idx[0][0]
    max_h = idx[0][len(idx[0])-1]
    min_w = min(idx[1])
    max_w = max(idx[1])

    return min_h, max_h, min_w, max_w

def get_crop_halves(img, min_h, max_h, min_w, max_w, size_x=256, size_y=256):

    img = np.array(img)
    
    dim = len(img)
    real_min_w = int(min_w*dim/size_x)
    real_max_w = int(max_w*dim/size_x)
    real_min_h = int(min_h*dim/size_y)
    real_max_h = int(max_h*dim/size_y)
    
    half_point = int(real_max_h*0.5)
    
    return [img[real_min_h:half_point, real_min_w:real_max_w], img[half_point:real_max_h, real_min_w:real_max_w]]

def get_groupC_quadrants(img, original, mask):
    
    [le, re, bs], [le_exits, re_exits, bs_exits] = get_regions(mask)
    quadrants = []
    quadrant_ids = []
    
    if le_exits:
        min_h, max_h, min_w, max_w = get_box(le)
        if min_h > len(le)*0.4:
            quadrant_ids += ['Q2']
            quadrants += [get_crop(original, min_h, max_h, min_w, max_w)]
        elif max_h < len(le)*0.6:
            quadrant_ids += ['Q1']
            quadrants += [get_crop(original, min_h, max_h, min_w, max_w)]
        else:
            quadrant_ids += ['Q1']
            quadrant_ids += ['Q2']
            quadrants += get_crop_halves(original, min_h, max_h, min_w, max_w)
        
    if re_exits:
        min_h, max_h, min_w, max_w = get_box(re)
        if min_h > len(re)*0.4:
            quadrant_ids += ['Q4']
            quadrants += [get_crop(original, min_h, max_h, min_w, max_w)]
        elif max_h < len(re)*0.6:
            quadrant_ids += ['Q3']
            quadrants += [get_crop(original, min_h, max_h, min_w, max_w)]
        else:
            quadrant_ids += ['Q3']
            quadrant_ids += ['Q4']
            quadrants += get_crop_halves(original, min_h, max_h, min_w, max_w)
        
    if bs_exits:
        min_h, max_h, min_w, max_w = get_box(bs)
        quadrants += [get_crop(original, min_h, max_h, min_w, max_w)]
        quadrant_ids += ['Q5']
        
    return quadrants, quadrant_ids

def five_cut(img_path, original, model):
    img = cv.imread(str(img_path))
    img = cv.resize(img, (256, 256))
    img = preprocessing(img)
    dim = max(len(original), len(original[0]))
    original = cv.resize(original, (dim, dim))
    mask = get_mask(img, model)
    quadrants, categories = get_groupC_quadrants(img, original, mask)
    
    return quadrants, categories

def four_cut(image, margin=0.1):
    
    height, width = image.shape[:2]
    
    start_row, start_col = int(0), int(0)
    mid1_row, mid1_col = int(height * (0.5+margin)), int(width * (0.5+margin))
    mid2_row, mid2_col = int(height * (0.5-margin)), int(width * (0.5-margin))
    end_row, end_col = int(height), int(width)

    
    if height >= width:
        q1 = image[start_row:mid1_row , :]
        q2 = image[mid2_row:end_row , :]
        q3 = np.array([])
        q4 = np.array([])

    else:
        q1 = image[start_row:mid1_row , start_col:mid1_col]
        q2 = image[mid2_row:end_row , start_col:mid1_col]

        q3 = image[start_row:mid1_row , mid2_col:end_col]
        q4 = image[mid2_row:end_row , mid2_col:end_col]
    
    return [q1, q2, q3, q4]

def is_empty_quadrant(img):
    if len(img) != 0:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
        min_area = len(binary)*len(binary[0])*0.40
        if int(np.sum(binary)/255) > min_area:
            return False
    return True



def gamma_correction(image, gamma=1.5):
    # Input validation
    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero.")

    normalized_image = image / 255.0
    corrected_image = np.power(normalized_image, gamma)
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    print (corrected_image[corrected_image>0].shape,np.min(corrected_image),np.max(corrected_image))

    # max_,min_=np.max(corrected_image),np.min(corrected_image)
    # corrected_image=((corrected_image-min_)/max_-min_)*255
    # Scale to [0, 255] and round

    return corrected_image


def apply_sharpening(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    kernel = kernel / np.sum(kernel)
    sharpened = cv.filter2D(image, -1, kernel)
    print("min and max sharpened", np.min(sharpened), np.max(sharpened))
    print(sharpened.dtype)
    # corrected_image[corrected_image > 255.0] = 255.0
    # corrected_image[corrected_image < 0.0] = 0.0
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    print("man and max sharpened_normalized", np.max(sharpened_normalized), np.min(sharpened_normalized))

    return sharpened_normalized