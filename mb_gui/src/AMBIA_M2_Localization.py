import os
import numpy as np
import cv2 as cv
from Switches_Dynamic import get_rootpath
import Switches_Static as st_switches
import Switches_Dynamic as dy_switches
import utils.allen_functions as allen
from utils.allen_functions import get_volume
import utils.img_processing as img_utils
#from utils.smart_functions import predict_Q_values_of_section, get_atlas_values
from utils.img_processing import normalize

img_utils.check_switch(st_switches)
prepath = get_rootpath()

if st_switches.section_classifier_on or st_switches.segmentation_1_20_on or st_switches.section_QL_on:
    from tensorflow.keras.models import load_model



if st_switches.atlas_type == "Adult":
    from regionscode.Regions_n_colors_adult import Region_names, create_regs_n_colors_per_sec_list
elif st_switches.atlas_type == "Rat":
    from regionscode.Regions_n_colors_rat import Region_names, create_regs_n_colors_per_sec_list

region_names = []
for item in Region_names:
    if item[:-3] not in region_names:
        region_names.append(item[:-3])

if st_switches.section_QL_on:
    vol, cm_left, cm_right, level_map, tree = get_volume(os.path.join(prepath, "models", "Allen_files","annotation_10_ccf17.nrrd"), region_names)


SL_classifier_model_path = os.path.join(prepath, "models", "single_regression_resnet50v2_v11")
QL_classifier_model_path = os.path.join(prepath, "models", "quadrant_regression_resnet101_v2")
seg20_model_path = os.path.join(prepath, "models", "seg_1_22_5")
sgc_model_path = os.path.join(prepath, "models",'sgc_v6')

if st_switches.section_classifier_on:
    SL_classifier_model = load_model(SL_classifier_model_path)
if st_switches.section_QL_on:
    QL_classifier_model = load_model(QL_classifier_model_path)
    print("QL model loaded")
    #sgc_model = load_model(sgc_model_path)
    print("sgc model loaded")
if st_switches.segmentation_1_20_on:
    seg20_model = load_model(seg20_model_path)
    print("1-20segmenter model loaded")
import utils.img_processing as img_utils


def predict_Q_values_of_section(img_path, sl_value, sgc_model, ql_model):
    q1_value, q2_value, q3_value, q4_value, q5_value = 0, 0, 0, 0, 0
    original_img = cv.imread(img_path)
    if sl_value < 24:
        # Send to registration
        section = {}
    elif sl_value > 83 and sl_value <= 103:
        # 5-Part Splitting with Segmentation CNN
        quads, categories = img_utils.five_cut(img_path, original_img, sgc_model)
        [q1_value, q2_value, q3_value, q4_value, q5_value] = get_quadrant_value_C(quads, categories, ql_model)

        section = {'img_path': str(img_path), 
                   'sl': sl_value,
                   'q1': q1_value, 
                   'q2': q2_value,
                   'q3': q3_value,
                   'q4': q4_value,
                   'q5': q5_value}

    else:
        # 4-Part Splitting
        # Split in 4
        quads = img_utils.four_cut(original_img)
        # Check for empty quadrants
        [q1_value, q2_value, q3_value, q4_value] = get_quadrant_value_A(quads, ql_model)

        section = {'img_path': str(img_path), 
               'sl': sl_value,
               'q1': q1_value, 
               'q2': q2_value,
               'q3': q3_value,
               'q4': q4_value,
               'q5': 0}
        
    return section
    

def get_quadrant_value_A(imgs, model, size_x=200, size_y=200):
    values = []
    for img in imgs:
        if img_utils.is_empty_quadrant(img):
            values += [0]
        else:
            X = cv.resize(img, (size_x, size_y))
            X = img_utils.preprocessing(X)
            value = model.predict(np.array([X]))[0][0]
            values += [value]
    return values


def get_quadrant_value_C(imgs, categories, model, size_x=200, size_y=200):
    Qs = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'] 
    values = []
    i = 0
    for Q in Qs:
        if Q not in categories:
            values += [0]
        else:
            X = cv.resize(imgs[i], (size_x, size_y))
            X = img_utils.preprocessing(X)
            value = model.predict(np.array([X]))[0][0]
            values += [value]
            i += 1
    return values



def get_atlas_values(atlas_Qs, sl_value):
    # Get position - use SL? Avg? Remove outliers?
    section ={'q1':atlas_Qs[0], 'q2':atlas_Qs[1], 'q3':atlas_Qs[2], 'q4':atlas_Qs[3], 'q5': 0, 'sl': sl_value}
    values = np.array([section['q1'], 
                       section['q2'], 
                       section['q3'], 
                       section['q4'],
                       section['q5']])
    
    values = values[values != 0]
    position = np.mean(values)
    
    # Get Alpha
    alphaA = section['q1'] - section['q3']
    alphaB = section['q2'] - section['q4']
    
    if abs(alphaA - alphaB) < 2:
        alpha = (alphaA + alphaB)/2
    else:
        avg1 = (section['q1'] + section['q3'])/2
        avg2 = (section['q2'] + section['q4'])/2
        if abs(avg1-section['sl']) < abs(avg2-section['sl']):
            alpha = alphaA
        else:
            alpha = alphaB
            
    # Get Beta
    betaA = section['q1'] - section['q2']
    betaB = section['q3'] - section['q4']
    
    if abs(betaA - betaB) < 2:
        beta = (betaA + betaB)/2
    else:
        avg1 = (section['q1'] + section['q2'])/2
        avg2 = (section['q3'] + section['q4'])/2
        if abs(avg1-section['sl']) < abs(avg2-section['sl']):
            beta = betaA
        else:
            beta = betaB
            
    allen_values = {'avg_position': position,
                    'alpha': alpha,
                    'beta': beta,
                    'sl': section['sl'],
                    'q1': section['q1'],
                    'q2': section['q2'],
                    'q3': section['q3'],
                    'q4': section['q4'],
                    'q5': section['q5']}
    
    return allen_values

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

def load_img(img):
    img = cv.imread(img)
    img = cv.resize(img, (200, 200))
    img = np.array([preprocessing(img)])
    return img

def predict_atlasnum(imgpath):    
    """
    This reads the image and uses the SL predictor model to predict the position of slice along AP axis,
    without considering the Angle. This is called the SL value
    """
    image = load_img(imgpath)
    pred_atlasnum = SL_classifier_model.predict(image)
    sl_value = int(pred_atlasnum[0][0])
    dy_switches.set_pred_atlas_sl_value(sl_value)
    return int(pred_atlasnum[0][0])


def predict_atlasQs(imgpath):
    #image = load_img(imgpath)
    pred_atlasquads = [38.1, 37.2, 39.3, 38.4]
    pred_Qs = [38.1, 37.2, 39.3, 38.4]
    if st_switches.section_QL_on and not st_switches.segmentation_1_20_on:
        sl_value = dy_switches.get_pred_sl_value()
        pred_atlasquads = predict_Q_values_of_section(imgpath, sl_value, QL_classifier_model, QL_classifier_model)    #predict_Q_values_of_section(img_path, sl_value, sgc_model, ql_model)
        print("used the QL predictor: ")
        
        pred_Qs = [round(pred_atlasquads['q1'],1), round(pred_atlasquads['q2'],1), round(pred_atlasquads['q3'],1), round(pred_atlasquads['q4'],1)]
        print(pred_Qs)
    return pred_Qs

def generate_tilted_atlas(atlas_Qs, sectionfolder):
    sl_value = dy_switches.get_pred_sl_value()
    allen_values = get_atlas_values(atlas_Qs, sl_value)
    tilted_atlas = allen.get_atlas(sl_value, vol, level_map, tree, 
                        alpha_number=allen_values['alpha'], 
                        beta_number=allen_values['beta'], 
                        cm_left=cm_right, 
                        cm_right=cm_right)
    
    tilted_atlas_path = os.path.join(sectionfolder, "tilted_atlas.png")
    cv.imwrite(tilted_atlas_path, tilted_atlas)
    return tilted_atlas_path


def smartfunc_segment_1_20(img_path, sectionfolder):
    img = cv.imread(img_path)
    img_resized = cv.resize(img, (256, 256))
    imggr = cv.imread(img_path, 0) #######optimize
    test_img = cv.resize(imggr, (256, 256))
    test_img_input=np.expand_dims(test_img, 0)
    test_img_norm = normalize(test_img_input, axis=1)
    prediction = (seg20_model.predict(test_img_norm))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    predicted_img2 = predicted_img * 40
    showimg = np.zeros((256,256,3))
    showimg[predicted_img==1] = (255,0,0)
    showimg[predicted_img==2] = (0,255,0)
    showimg[predicted_img==3] = (255,255,0)
    showimg[predicted_img==4] = (0,0,255)
    showimg[predicted_img==5] = (200,200,200)
    showimg[predicted_img==6] = (255,0,255)
    img_resized = np.asarray(img_resized, np.float64)
    segmented_img_rgb = np.asarray(showimg, np.float64)
    segmented_img_path = os.path.join(sectionfolder, "segmented_atlas.png")
    
    print("types:", type(img_resized), type(showimg))
    segmented_img_gr = predicted_img * 40
    
    cv.imwrite(segmented_img_path, segmented_img_rgb)
    showimg2 = cv.addWeighted(img_resized, 1, showimg, 0.4, 0)
    return segmented_img_path, showimg2