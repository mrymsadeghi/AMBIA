import os
import numpy as np
import matplotlib.pyplot as plt
import utils.allen_functions as allen
#from tensorflow.keras.models import load_model
#from tensorflow.saved_model import load
from time import time
#import atlas_generator as allen
from colormap import rgb2hex, hex2rgb
from allensdk.core.reference_space_cache import ReferenceSpaceCache, ReferenceSpace
import Switches_Static as st_switches
import Switches_Dynamic as dy_switches
import onnxruntime as ort
import cv2
import utils.img_processing as smart_utils 
rootpath = dy_switches.get_rootpath()
#SL prediction model
try :
    sl_model_path=os.path.join(rootpath,"mb_gui/models","single_regression_resnet50v2_v11.onnx")
    sl_model = ort.InferenceSession(sl_model_path)
    sl_model_input_name = sl_model.get_inputs()[0].name
    sl_model_output_name = sl_model.get_outputs()[0].name

    #Segmentation models
    sgc_model_path=os.path.join(rootpath,"mb_gui/models","SGC_V6.onnx")
    sgc_model=ort.InferenceSession(sgc_model_path)
    sgc_model_input_name=sgc_model.get_inputs()[0].name
    sgc_model_output_name=sgc_model.get_outputs()[0].name

    #QL Prediction model
    ql_model_path=os.path.join(rootpath,"mb_gui/models","quadrant_regression_resnet101_v2.onnx")
    ql_model=ort.InferenceSession(ql_model_path)
    ql_model_input_name=ql_model.get_inputs()[0].name
    ql_model_output_name=ql_model.get_outputs()[0].name
except  ort.capi.onnxruntime_pybind11_state.NoSuchFile :
    print ("""Smart models were not found. You will not be able to use AMBIA smart functions.
Please contact the following to receive the model files:  "amir.bakhtiary@ut.ac.ir" """)

annotations_file_path = os.path.join(rootpath, 'accessories','annotation_10_ccf17.nrrd')
#print (annotations_file_path,"anot")
region_names = dy_switches.get_region_names_for_alas_generation()


#util functions


def standardize(x):
    x = np.array(x, dtype='float32')
    x -= np.min(x)
    x /= np.percentile(x, 98)
    x[x > 1] = 1
    return x

def preprocessing_SL(img):
    image = np.array(img)   
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = np.zeros_like(image)
    image[:,:,0] = gray
    image[:,:,1] = gray
    image[:,:,2] = gray
    image = standardize(image)
    return image

def five_cut(original):
    img = cv2.resize(original, (256, 256))
    img = preprocessing_SL(img)


    dim = max(len(original), len(original[0]))
    original = cv2.resize(original, (dim, dim))
    #print (img.shape)
    pred_mask=sgc_model.run([sgc_model_output_name], {sgc_model_input_name: np.expand_dims(img,0).astype("float32")})[0]
    pred_mask = np.argmax(pred_mask, axis=-1)
    mask = np.expand_dims(pred_mask,-1)[0]#[..., tf.newaxis]
    #print (mask.shape)
    quadrants, categories = smart_utils.get_groupC_quadrants( original, mask)
    return quadrants, categories


def get_SL_value_smart(img,size_y=200, size_x=200):
    img_resize = cv2.resize(img, (size_y, size_x))
    sl_img = preprocessing_SL(img_resize)
    sl_img=np.expand_dims(sl_img,0)
    sl_value = sl_model.run([sl_model_output_name], {sl_model_input_name: sl_img})[0][0]
    return sl_value[0]

def crop_image_for_SL(img):
    original_img = img.copy()
    img = preprocessing_SL(original_img)
    ret, binary = cv2.threshold(img,0.1, 1, cv2.THRESH_BINARY)
    
    idx = np.where(binary==1)
    min_h = idx[0][0]
    max_h = idx[0][len(idx[0])-1]
    min_w = min(idx[1])
    max_w = max(idx[1])
        
    return original_img[min_h:max_h, min_w:max_w, :]

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
    imgs=[]
    if len(img) != 0:
        ret, binary = cv2.threshold(img[:,:,0], 0.2, 1, cv2.THRESH_BINARY)
        min_area = len(binary)*len(binary[0])*0.40
        imgs += [binary]
        if np.sum(binary) > min_area:
            return False
    return True


def get_quadrant_value_A(imgs, size_x=200, size_y=200):
    values = []
    for img in imgs:
        X = cv2.resize(img, (size_x, size_y))
        X = preprocessing_SL(X)
        if is_empty_quadrant(X):
            values += [0]
        else:
            #needs changing
            value=ql_model.run([ql_model_output_name], {ql_model_input_name: np.array([X]).astype("float32")})[0][0]
            #value = ql_model.predict(np.array([X]))[0][0]
            values += [value[0]]
    return values


def get_quadrant_value_C(imgs, categories, model, size_x=200, size_y=200):
    Qs = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'] 
    values = []
    i = 0
    for Q in Qs:
        if Q not in categories:
            values += [0]
        else:
            X = cv2.resize(imgs[i], (size_x, size_y))
            X = preprocessing_SL(X)
            value = ql_model.run([ql_model_output_name], {ql_model_input_name: np.array([X]).astype("float32")})[0][0]
            values += [value[0]]
            i += 1
    return values

def get_QL_values_smart(img_path, img, sl_value,):
    q1_value, q2_value, q3_value, q4_value, q5_value = 0, 0, 0, 0, 0
    if sl_value < 24:
        # Send to registration
        section = {'sl': sl_value}
    elif sl_value > 83 and sl_value <= 103:
        # 5-Part Splitting with Segmentation CNN
        quads, categories = five_cut(img)
        [q1_value, q2_value, q3_value, q4_value, q5_value] = get_quadrant_value_C(quads, categories, ql_model)

        section = {'img_path': str(img_path), 
                   'sl': sl_value,
                   'q1': q1_value, 
                   'q2': q2_value,
                   'q3': q3_value,
                   'q4': q4_value,
                   'q5': q5_value,
                   'img': img}

    else:
        # 4-Part Splitting
        # Split in 4
        #img=crop_image_for_SL(img)
        quads = four_cut(img)
        # Check for empty quadrants
        [q1_value, q2_value, q3_value, q4_value] = get_quadrant_value_A(quads)

        section = {'img_path': str(img_path), 
               'sl': sl_value,
               'q1': q1_value, 
               'q2': q2_value,
               'q3': q3_value,
               'q4': q4_value,
               'q5': 0,
               'img': img}
        
    return section
    
def find_outlier(q1, q2, q3, q4, avg):
    q1_offset = abs(avg - q1)
    q2_offset = abs(avg - q2)
    q3_offset = abs(avg - q3)
    q4_offset = abs(avg - q4)

    offsets = [q1_offset, q2_offset, q3_offset, q4_offset]
    index = offsets.index(max(offsets))
    
    if index == 0:
        return 'Q1'
    elif index == 1:
        return 'Q2'
    elif index == 2:
        return 'Q3'
    else:
        return 'Q4'


# calculating atlases
def calculate_atlas_values(section_stats): 
    """if not sl_value:
        sl_value=sum(section_q_vals)/4   """      
    #section_stats = {'sl': sl_value,'q1': section_q_vals[0], 'q2': section_q_vals[1], 'q3': section_q_vals[2], 'q4': section_q_vals[3], 'q5': st_switches.q5value }
    values = np.array([section_stats['q1'], 
                       section_stats['q2'], 
                       section_stats['q3'], 
                       section_stats['q4'],
                       section_stats['q5']]) 

    values = values[values != 0]
    position = np.mean(values)
    # Get Alpha
    alphaA = section_stats['q1'] - section_stats['q3']
    alphaB = section_stats['q2'] - section_stats['q4']
    
    if abs(alphaA - alphaB) < 2:
        alpha = (alphaA + alphaB)/2
    else:
        avg1 = (section_stats['q1'] + section_stats['q3'])/2
        avg2 = (section_stats['q2'] + section_stats['q4'])/2
        if abs(avg1-section_stats['sl']) < abs(avg2-section_stats['sl']):
            alpha = alphaA
        else:
            alpha = alphaB     
    # Get Beta
    betaA = section_stats['q1'] - section_stats['q2']
    betaB = section_stats['q3'] - section_stats['q4']
    
    if abs(betaA - betaB) < 2:
        beta = (betaA + betaB)/2
    else:
        avg1 = (section_stats['q1'] + section_stats['q2'])/2
        avg2 = (section_stats['q3'] + section_stats['q4'])/2
        if abs(avg1-section_stats['sl']) < abs(avg2-section_stats['sl']):
            beta = betaA
        else:
            beta = betaB
    section_stats['alpha'] = alpha
    section_stats['beta'] = beta 
    section_stats['avg_position'] = position       
   
    return section_stats

def get_atlas_values_smart(section):
    
    threshold = 1.75
    
    # Get position - use SL? Avg? Remove outliers?
    values = np.array([section['q1'], 
                       section['q2'], 
                       section['q3'], 
                       section['q4'],
                       section['q5']])
    
    values = values[values != 0]
    avg_position = np.mean(values)
    
    # If there's one missing value -> calculate the angle with the others
    # Only one quadrant (Q4)
    """
    Changed this part to one line
    
    if section['q1'] == 0 and section['q2'] == 0 and section['q3'] == 0:
        alpha = 0
        beta = 0
    # Only one quadrant (Q3)
    elif section['q1'] == 0 and section['q2'] == 0 and section['q4'] == 0:
        alpha = 0
        beta = 0
    # Only one quadrant (Q2)
    elif section['q1'] == 0 and section['q3'] == 0 and section['q4'] == 0:
        alpha = 0
        beta = 0
    # Only one quadrant (Q1)
    elif section['q2'] == 0 and section['q3'] == 0 and section['q4'] == 0:
        alpha = 0
        beta = 0
    """
    # 3 quadrants missing
    quads = np.array([section['q1'], section['q2'], section['q3'], section['q4']])
    if np.count_nonzero(quads)<2:
        alpha, beta = 0, 0
        
    # Top missing
    elif section['q1'] == 0 and section['q2'] == 0:
        alpha = 0
        beta = section['q3'] - section['q4']
    # Bottom missing
    elif section['q3'] == 0 and section['q4'] == 0:
        alpha = 0
        beta = section['q1'] - section['q2']
    # Left missing
    elif section['q1'] == 0 and section['q3'] == 0:
        alpha = section['q2'] - section['q4']
        beta = 0
    # Right missing
    elif section['q2'] == 0 and section['q4'] == 0:
        alpha = section['q1'] - section['q3']
        beta = 0
    # Q1 missing
    elif section['q1'] == 0:
        alpha = section['q2'] - section['q4']
        beta = section['q3'] - section['q4']
    # Q2 missing
    elif section['q2'] == 0:
        alpha = section['q1'] - section['q3']
        beta = section['q3'] - section['q4']
    # Q3 missing
    elif section['q3'] == 0:
        alpha = section['q2'] - section['q4']
        beta = section['q1'] - section['q2']
    # Q4 missing
    elif section['q4'] == 0:
        alpha = section['q1'] - section['q3']
        beta = section['q1'] - section['q2']
        
    # if no quadrant is missing:
    else:
        '''
        if (abs(alphaA - alphaB) < 1.75) AND (abs(betaA - betaB) < 1.75):
            alpha = (alphaA + alphaB)/2
        else:
            discard the one quadrant value which is the furthest from the SL or the avgposition (still not sure which one!)
        '''
        # Get Sub-Alphas
        alphaA = section['q1'] - section['q3']
        alphaB = section['q2'] - section['q4']

        # Get Sub-Betas
        betaA = section['q1'] - section['q2']
        betaB = section['q3'] - section['q4']

        if (abs(alphaA - alphaB) < threshold) and (abs(betaA - betaB) < threshold):
            alpha = (alphaA + alphaB)/2
            beta = (betaA + betaB)/2
        else:
            q = find_outlier(section['q1'], section['q2'], section['q3'], section['q4'], avg_position)
            if q == 'Q1':
                alpha = section['q2'] - section['q4']
                beta = section['q3'] - section['q4']
                avg_position = np.mean([section['q2'],section['q3'],section['q4']])   #avgposition needs to be updated after removing the outlier

            elif q == 'Q2':
                alpha = section['q1'] - section['q3']
                beta = section['q3'] - section['q4']
                avg_position = np.mean([section['q1'],section['q3'],section['q4']])

            elif q == 'Q3':
                alpha = section['q2'] - section['q4']
                beta = section['q1'] - section['q2']
                avg_position = np.mean([section['q1'],section['q2'],section['q4']])

            else:
                alpha = section['q1'] - section['q3']
                beta = section['q1'] - section['q2']
                avg_position = np.mean([section['q1'],section['q2'],section['q3']])

    
    # Old System
    '''
    
    if abs(alphaA - alphaB) < 2:
        alpha = (alphaA + alphaB)/2
    else:
        avg1 = (section['q1'] + section['q3'])/2
        avg2 = (section['q2'] + section['q4'])/2
        if abs(avg1-section['sl']) < abs(avg2-section['sl']):
            alpha = alphaA
        else:
            alpha = alphaB
    
    if abs(betaA - betaB) < 2:
        beta = (betaA + betaB)/2
    else:
        avg1 = (section['q1'] + section['q2'])/2
        avg2 = (section['q3'] + section['q4'])/2
        if abs(avg1-section['sl']) < abs(avg2-section['sl']):
            beta = betaA
        else:
            beta = betaB
            
    '''
            
    allen_values = {'avg_position': round(avg_position,2),
                    'alpha': round(alpha,2),
                    'beta': round(beta,2),
                    'sl': round(section['sl'],2),
                    'q1': round(section['q1'],2),
                    'q2': round(section['q2'],2),
                    'q3': round(section['q3'],2),
                    'q4': round(section['q4'],2),
                    'q5': round(section['q5'],2)}
    
    
    return allen_values

def get_atlas_values_smart_gc(section):
    
    values = np.array([section['q1'], 
                   section['q2'], 
                   section['q3'], 
                   section['q4'],
                   section['q5']])

    values = values[values != 0]
    position = np.mean(values)
    
    missing_left_ear = False
    missing_right_ear = False
    
    if section['q1'] != 0 and section['q2'] != 0:
        left_position = (section['q1'] + section['q2'])/2
        beta_left = section['q1'] - section['q2']
    elif section['q1'] + section['q2'] > 0:
        left_position = section['q1'] + section['q2']
        beta_left = 0
    else:
        beta_left=0
        left_position=0
        missing_left_ear = True
    #print (beta_left,missing_left_ear)
    if section['q3'] != 0 and section['q4'] != 0:
        right_position = (section['q3'] + section['q4'])/2
        beta_right = section['q3'] - section['q4']
    elif section['q3'] + section['q4'] > 0:
        right_position = section['q3'] + section['q4']
        beta_right = 0
    else:
        beta_right=0
        right_position=0
        missing_right_ear = True
    #print (beta_right,missing_right_ear)
    # Both ears missing -> use Q5    
    if missing_left_ear and missing_right_ear:
        left_position = section['q5']
        right_position = section['q5']
        beta_left = 0
        beta_right = 0
        
    # One ear is missing -> use values of the other    
    """elif missing_left_ear:
        left_position = right_position
        beta_left = beta_right
        
    else:
        right_position = left_position
        beta_right = beta_left"""
    
    allen_values = {'avg_position': position,
                'left_position': left_position,
                'right_position': right_position,
                'beta_left': beta_left,
                'beta_right': beta_right,
                'sl': section['sl'],
                'q1': section['q1'],
                'q2': section['q2'],
                'q3': section['q3'],
                'q4': section['q4'],
                'q5': section['q5']}
    
    return allen_values

def prepare_atlas_generation():
    vol, cm_left, cm_right, level_map, tree = allen.get_volume(annotations_file_path, region_names)
    return vol, cm_left, cm_right, level_map, tree


def generate_Qs_atlas(section_q_values,smart=False):
    #vol, cm_left, cm_right, level_map, tree = prepare_atlas_generation()
    vol, cm_left, cm_right, level_map, tree,level_map_id_to_name = allen.get_volume(annotations_file_path, region_names)


    if section_q_values['q5']>1:
        section_stats=get_atlas_values_smart_gc(section_q_values)
        #print (section_stats)
        atlas_img,atlas_img_by_id = allen.get_off_plane_atlas(
                        left_slice_number= section_stats['left_position'], 
                        right_slice_number= section_stats['right_position'],
                        q5_slice_number = section_stats['q5'],
                        vol = vol, 
                        level_map = level_map, 
                        tree = tree, 
                        beta_left=section_stats['beta_left'], 
                        beta_right=section_stats['beta_right'], 
                        cm_left=cm_right, 
                        cm_right=cm_right)
        return atlas_img,atlas_img_by_id,level_map_id_to_name,cm_right
        
    #else :
    if section_q_values["sl"]<1:
            tmp=sum([section_q_values["q1"],section_q_values["q2"],
                                          section_q_values["q3"],section_q_values["q4"]])
            section_q_values["sl"]=tmp/4
    section_stats = get_atlas_values_smart(section_q_values)
    #else:
    """    if section_stats["sl"]<1:
            tmp=sum([section_stats["q1"],section_stats["q2"],
                                          section_stats["q3"],section_stats["q4"]])
            section_stats["sl"]=tmp/4
        section_stats = calculate_atlas_values(section_q_values)"""
    alpha=section_stats['alpha'] 
    beta=section_stats['beta'] 
    position=section_stats['avg_position']
    atlas_img,atlas_img_by_id = allen.get_atlas(position, vol,level_map, tree, alpha_number=alpha, beta_number=beta, cm_left=cm_right, cm_right=cm_right,print_regions=True)

    return atlas_img,atlas_img_by_id,level_map_id_to_name,cm_right

def generate_Qs_smart(img_path=None):

    print ("Starting smart function")
    img=cv2.imread(img_path)
    cropped_image = crop_image_for_SL(img)
    sl_value = get_SL_value_smart(cropped_image)
    section_q_values=get_QL_values_smart(img_path, cropped_image, sl_value)

    return section_q_values
if __name__=="__main__":#This was for testing
    
    """stat=generate_Qs_smart("t2.jpg")
    atlas_img,atlas_img_by_id,level_map_id_to_name,cm_right=generate_Qs_atlas(stat,True)
    cv2.imwrite("test.png",atlas_img)"""

