import os
import numpy as np
import matplotlib.pyplot as plt
import utils_allen as allen
#from tensorflow.keras.models import load_model
#from tensorflow.saved_model import load
from time import time
import atlas_generator as allen
from colormap import rgb2hex, hex2rgb
from allensdk.core.reference_space_cache import ReferenceSpaceCache, ReferenceSpace
import Switches_Static as st_switches
import Switches_Dynamic as dy_switches


rootpath = dy_switches.get_root_path()
annotations_file_path = os.path.join(rootpath, 'accessories','annotation_10_ccf17.nrrd')
region_names = dy_switches.get_region_names_for_alas_generation()


def calculate_atlas_values(section_q_vals, sl_value):          
    section_stats = {'sl': sl_value,'q1': section_q_vals[0], 'q2': section_q_vals[1], 'q3': section_q_vals[2], 'q4': section_q_vals[3], 'q5': st_switches.q5value }
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



def prepare_atlas_generation():
    vol, cm_left, cm_right, level_map, tree = allen.get_volume(annotations_file_path, region_names)
    return vol, cm_left, cm_right, level_map, tree


def generate_Qs_atlas(section_q_values):
    vol, cm_left, cm_right, level_map, tree = prepare_atlas_generation()
    sl_value = dy_switches.get_pred_sl_value()
    section_stats = calculate_atlas_values(section_q_values, sl_value)
    atlas_img = allen.get_atlas(79, vol,level_map, tree, alpha_number=0.2, beta_number=0, cm_left=cm_right, cm_right=cm_right)
    return atlas_img

