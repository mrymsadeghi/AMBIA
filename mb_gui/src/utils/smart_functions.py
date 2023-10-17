import numpy as np
import cv2 as cv
import img_processing as img_utils


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