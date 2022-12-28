import os
import pandas as pd
import nrrd
import numpy as np
import random
from colormap import hex2rgb
from allensdk.core.reference_space_cache import ReferenceSpaceCache, ReferenceSpace
import json 
from Switches_Dynamic import get_rootpath


prepath = get_rootpath()
allen_files_path = os.path.join(prepath, "models", "Allen_files")
MANIFEST_PATH = os.path.join(allen_files_path, 'manifest.json')


def get_volume(volume_path, region_names0=None, json_color_code=None):
    
    global region_names
    # Get hierarchy and list of regions
    rspc = ReferenceSpaceCache(10, 'annotation/ccf_2017', manifest=MANIFEST_PATH)
    tree = rspc.get_structure_tree(structure_graph_id=1) 
    
    if region_names0 == None:
        
        region_names = ['FRP', 'MO', 'SS', 'GU', 'VISC', 'AUD', 'VIS', 'ACA', 'PL', 'ILA', 'ORB', 'AI', 'RSP', 'PTLp', 'TEa', 'PERI', 'ECT', 'MOB', 'AOB', 'AON', 'TT', 'DP', 'PIR', 'NLOT', 'COA', 'PAA', 'TR', 'HIP', 'RHP', 'CLA', 'EP', 'LA', 'BLA', 'BMA', 'PA', 'STRd', 'STRv', 'LSX', 'sAMY', 'PALd', 'PALv', 'PALm', 'PALc', 'VENT', 'SPF', 'SPA', 'PP', 'GENd', 'LAT', 'ATN', 'MED', 'MTN', 'ILM', 'RT', 'GENv', 'EPI', 'PVZ', 'PVR', 'MEZ', 'LZ', 'ME', 'SCs', 'IC', 'NB', 'SAG', 'PBG', 'MEV', 'SNr', 'VTA', 'RR', 'MRN', 'SCm', 'PAG', 'PRT', 'CUN', 'RN', 'III', 'EW', 'IV', 'VTN', 'AT', 'LT', 'SNc', 'PPN', 'RAmb', 'P-sen', 'P-mot', 'P-sat', 'MY-sen', 'MY-mot', 'MY-sat', 'LING', 'CENT', 'CUL', 'DEC', 'FOTU', 'PYR', 'UVU', 'NOD', 'SIM', 'AN', 'PRM', 'COPY', 'PFL', 'FL', 'FN', 'IP', 'DN', 'fiber tracts', 'VS', 'OLF', 'CTX', 'HY', 'TH', 'MB', 'P', 'MY', 'CB']
        
    else:
        region_names = region_names0

    vol, metaVol = nrrd.read(volume_path)
    
    if json_color_code != None:
        cm_left, cm_right, level_map = import_colors_json(json_color_code, tree, region_names)
        
    else:
        cm_left, cm_right, level_map = new_colors(region_names, tree)
    
    return vol, cm_left, cm_right, level_map, tree

def get_atlas(slice_number, vol, level_map, tree, alpha_number=0, beta_number=0, cm_left={}, cm_right={}, print_regions=False):
    
    # alpha = right slice number - left slice number
    # beta = bottom slice number - top slice number

    slice_number = slice_number * 10 - 7
    alpha = alpha_number * 10  
    beta = beta_number * 10
    
    interval_alpha = vol.shape[2]*0.75 - vol.shape[2]*0.25
    pace_alpha = alpha/interval_alpha
    initial_alpha = -alpha

    # 200h == sup_slice_number
    # 600h == inf_slice_number
    interval_beta = vol.shape[1]*0.75 - vol.shape[1]*0.25
    pace_beta = beta/interval_beta
    initial_beta = -beta

    img_left = []
    for i in range(vol.shape[1]):
        #print(int(initial + pace*i))
        row = []
        for j in range(int(vol.shape[2]/2)):
            number = int(slice_number + initial_beta + initial_alpha + pace_beta*i + pace_alpha*j)
            if number < 0:
                number = 0
            if number >= vol.shape[0]:
                number = vol.shape[0] - 1
            row += [vol[number][i][j]]
        img_left += [row]
    values = np.unique(img_left)
    img_left = np.array(img_left).astype('float64')
    img_left = np.reshape([cm_left[point] for point in img_left.flat], list(img_left.shape) + [3]).astype(np.uint8)

    img_right = []
    for i in range(vol.shape[1]):
        row = []
        for j in range(int(vol.shape[2]/2), vol.shape[2]):
            number = int(slice_number + initial_beta + initial_alpha + pace_beta*i + pace_alpha*j)
            if number < 0:
                number = 0
            if number >= vol.shape[0]:
                number = vol.shape[0] - 1
            row += [vol[number][i][j]]
        img_right += [row]
    img_right = np.array(img_right).astype('float64')
    img_right = np.reshape([cm_right[point] for point in img_right.flat], list(img_right.shape) + [3]).astype(np.uint8)

    img = [np.concatenate((img_left[i], img_right[i])) for i in range(len(img_right))]
    
    
    if print_regions:
        values = np.delete(values, np.where(values == 0))
        acrs = [tree.get_structures_by_id([level_map[v]])[0]['acronym'] for v in values]
        #print(set(acrs))
    
    return np.array(img)

def get_off_plane_atlas(slice_number, vol, level_map, tree, alpha_left=0, alpha_right=0, beta_left=0, beta_right=0, cm_left={}, cm_right={}, print_regions=False):
    
    # alpha = right slice number - left slice number
    # beta = bottom slice number - top slice number

    slice_number = slice_number * 10 - 7
    alpha_left = alpha_left * 10  
    beta_left = beta_left * 10
    alpha_right = alpha_right * 10  
    beta_right = beta_right * 10
        
    interval_beta_left = vol.shape[1]*0.75 - vol.shape[1]*0.25
    pace_beta_left = (beta_left)/interval_beta_left
    initial_beta_left = -beta_left
    
    interval_beta_right = vol.shape[1]*0.75 - vol.shape[1]*0.25
    pace_beta_right = (beta_right)/interval_beta_right
    initial_beta_right = -beta_right

    left_ear_img = vol[int(slice_number + alpha_left)]
    right_ear_img = vol[int(slice_number + alpha_right)]
    brain_stem_img = vol[int(slice_number)]
    
    left_ear_img = []
    for i in range(vol.shape[1]):
        row = []
        for j in range(vol.shape[2]):
            row += [vol[int(slice_number + alpha_left + pace_beta_left*i)][i][j]]
        left_ear_img += [row]
        
    right_ear_img = []
    for i in range(vol.shape[1]):
        row = []
        for j in range(vol.shape[2]):
            row += [vol[int(slice_number + alpha_right + pace_beta_right*i)][i][j]]
        right_ear_img += [row]

    brain_stem_id = [343, 960, 967, 784, 1000, 512, 73, 1092] # BS ID

    brain_stem_img = np.array(brain_stem_img)
    brain_stem_img = np.reshape([in_parents(point, brain_stem_id, tree) for point in brain_stem_img.flat], list(brain_stem_img.shape))

    left_brain_stem = brain_stem_img
    right_brain_stem = brain_stem_img

    left_ear_img = np.array(left_ear_img)
    left_ear_img = np.reshape([not_in_parents(point, brain_stem_id, tree) for point in left_ear_img.flat], list(left_ear_img.shape))

    right_ear_img = np.array(right_ear_img)
    right_ear_img = np.reshape([not_in_parents(point, brain_stem_id, tree) for point in right_ear_img.flat], list(right_ear_img.shape))

    left_brain_stem = np.array(left_brain_stem)

    right_brain_stem = np.array(right_brain_stem)

    img_left = []
    for i in range(len(left_ear_img)):
        row = []
        for j in range(len(left_ear_img[0])):
            if left_ear_img[i][j] == 0 and left_brain_stem[i][j] != 0:
                row += [left_brain_stem[i][j]]
            else: 
                row += [left_ear_img[i][j]]
        img_left += [row]

    values = np.unique(img_left)
    img_left = np.array(img_left).astype('float64')
    img_left = np.reshape([cm_left[point] for point in img_left.flat], list(img_left.shape) + [3]).astype(np.uint8)
    img_left = np.array([row[:int(img_left.shape[1]/2)] for row in img_left])

    img_right = []
    for i in range(len(right_ear_img)):
        row = []
        for j in range(len(right_ear_img[0])):
            if right_ear_img[i][j] == 0 and right_brain_stem[i][j] != 0:
                row += [right_brain_stem[i][j]]
            else: 
                row += [right_ear_img[i][j]]
        img_right += [row]

    img_right = np.array(img_right).astype('float64')
    img_right = np.reshape([cm_right[point] for point in img_right.flat], list(img_right.shape) + [3]).astype(np.uint8)
    img_right = np.array([row[int(img_right.shape[1]/2):] for row in img_right])

    img = [np.concatenate((img_left[i], img_right[i])) for i in range(len(img_right))]

    if print_regions:
        values = np.delete(values, np.where(values == 0))
        acrs = [tree.get_structures_by_id([level_map[v]])[0]['acronym'] for v in values]
        #print(set(acrs))
    
    return np.array(img)


def import_colors_json(json_color_code, tree, region_names):
    
    atlas_values = list(tree.get_id_acronym_map().values())
    
    # Create color correspondence map and get list of ids to assign color
    level_map = create_map(tree, atlas_values, region_names)
    
    with open(json_color_code, 'r') as f:
        data = json.load(f)

    cm_left = {}
    for obj in data[0]['left']:
        color = obj['color']
        if color[0] == '#':
            cm_left[obj['id']] = hex2rgb(color)
        else:
            cm_left[obj['id']] = [int(i) for i in color[1:-1].split(', ')]


    cm_right = {}
    for obj in data[1]['right']:
        color = obj['color']
        if color[0] == '#':
            cm_right[obj['id']] = hex2rgb(color)
        else:
            cm_right[obj['id']] = [int(i) for i in color[1:-1].split(', ')]

    cm_left[0] = hex2rgb('#000000')
    cm_right[0] = hex2rgb('#000000')
    
    return cm_left, cm_right, level_map

def new_colors(region_names, tree):
    
    atlas_values = list(tree.get_id_acronym_map().values())
    
    # Create color correspondence map and get list of ids to assign color
    level_map = create_map(tree, atlas_values, region_names)
    
    # List of IDs of the regions that need a color assigned (region_names and roots)
    ids_to_color = list(set(level_map.values())) 
    
    # Gets the color for the specified regions in region names
    specified_cm_left, specified_cm_right = assign_colors(ids_to_color)

    # Attribute a color to EVERY region at any level
    cm_left = {}
    cm_right = {}

    for i in range(len(atlas_values)):
        cm_left[atlas_values[i]] = specified_cm_left[level_map[atlas_values[i]]]
        cm_right[atlas_values[i]] = specified_cm_right[level_map[atlas_values[i]]]
        
    cm_left[0] = [0, 0, 0]
    cm_right[0] = [0, 0, 0]
        
    return cm_left, cm_right, level_map

def assign_colors(atlas_values):
    rgb_values_left = []
    for _ in range(len(atlas_values)): 
        while 1:
            r = int(random.uniform(0, 256))
            g = int(random.uniform(0, 256))
            b = int(random.uniform(0, 256))
            if [r, g, b] not in rgb_values_left:
                rgb_values_left.append([r, g, b]) 
                break

    np_rgb_values_left = np.array(rgb_values_left)
    cm_left = {}

    for i in range(len(atlas_values)):
        if atlas_values[i] == 0:
            cm_left[0] = [0, 0, 0] 
        elif atlas_values[i] == 997:
            cm_left[atlas_values[i]] = [255, 0, 0]
        else:
            color = np_rgb_values_left[i].tolist()
            cm_left[atlas_values[i]] = color
            
    rgb_values_right = []

    for _ in range(len(atlas_values)): 
        while 1:
            r = int(random.uniform(0, 256))
            g = int(random.uniform(0, 256))
            b = int(random.uniform(0, 256))
            if [r, g, b] not in rgb_values_left and [r, g, b] not in rgb_values_right:
                rgb_values_right.append([r, g, b]) 
                break

    np_rgb_values_right = np.array(rgb_values_right)
    cm_right = {}

    for i in range(len(atlas_values)):
        if atlas_values[i] == 0:
            cm_right[0] = [0, 0, 0] 
        elif atlas_values[i] == 997:
            cm_right[atlas_values[i]] = [0, 255, 0] 
        else:
            color = np_rgb_values_right[i].tolist()
            cm_right[atlas_values[i]] = color
            
    cm_right[0] = [0, 0, 0] 
    cm_left[0] = [0, 0, 0] 
    
    return cm_left, cm_right

def create_map(tree, atlas_values, region_names):
 
    # Create correspondence map to relate regions not in region_names to the closest parent in region_names
    level_map = {}
    for value in atlas_values:
        if value != 0:
            if value != 997: #root
                if tree.get_structures_by_id([value])[0]['acronym'] in region_names:
                    level_map[value] = value
                else:
                    parents = tree.parents([value])[0]['structure_id_path']
                    i = len(parents) - 1
                    while i >= 0:
                        if tree.get_structures_by_id([parents[i]])[0]['acronym'] in region_names:
                            level_map[value] = parents[i]
                            i = -2
                        i -= 1
                    if i == -1:
                        level_map[value] = 997
    
    level_map[997] = 997 
    level_map[0] = 0    
            
    return level_map

# Get list of specified regions, with respective parents and colors

def get_color_code_simple(cm):
    
    # Get hierarchy and list of regions
    rspc = ReferenceSpaceCache(10, 'annotation/ccf_2017', manifest=MANIFEST_PATH)
    tree = rspc.get_structure_tree(structure_graph_id=1) 
    
    objs = []
    for v in cm:
        if v != 0:
            acr = tree.get_structures_by_id([v])[0]['acronym']
            if acr in region_names:
                color = cm[v][1:]
                parents_ids = tree.get_structures_by_id([v])[0]['structure_id_path']
                parents = []
                for parent_id in parents_ids:
                    parents.append(tree.get_structures_by_id([parent_id])[0]['acronym'])

                obj = parents

                obj.append(color)
                obj.append(hex2rgb(cm[v]))

                obj = tuple(obj)

                objs.append(obj)
            
    return objs

def save_color_code(json_path, cm_left, cm_right):
        
    # Get hierarchy and list of regions
    rspc = ReferenceSpaceCache(10, 'annotation/ccf_2017', manifest=MANIFEST_PATH)
    tree = rspc.get_structure_tree(structure_graph_id=1) 
    
    # Get the colors and other attributes of regions based on the CM
    color_json = [{'left': cm_to_json(tree, cm_left)}, {'right': cm_to_json(tree, cm_right)}]

    with open(json_path, 'w') as f:
        json.dump(color_json, f)
        
    return True

def cm_to_json(tree, cm):
    
    objs = []
    for v in cm:
        if v != 0:
            acr = tree.get_structures_by_id([v])[0]['acronym']
            color = str(cm[v])
            
            parents_ids = tree.get_structures_by_id([v])[0]['structure_id_path']
            parents = []
            for parent_id in parents_ids:
                parents.append(tree.get_structures_by_id([parent_id])[0]['acronym'])
                
            visual = tree.get_structures_by_id([level_map[v]])[0]['acronym']
            
            obj = {
                'id': v,
                'name': acr,
                'color': color,
                'parents': parents,
                'visual': visual
            }
            
            objs.append(obj)
            
    return objs

def not_in_parents(value, regions, tree):
    if value == 0:
        return 0
    if value == 997:
        return 997
    
    for region_id in regions:
        
        if region_id in tree.parents([int(value)])[0]['structure_id_path'] or value == region_id:
            return 0
        
    return value

def in_parents(value, regions, tree):
    if value == 0:
        return 0
    if value == 997:
        return 997
    
    for region_id in regions:
        
        if region_id in tree.parents([int(value)])[0]['structure_id_path'] or value == region_id:
            return value
        
    return 0

def reset_count(regions):
    count = {}
    existing = {}
    for region in regions:
        count[region] = 0
        existing[region] = False
    return count, existing


def high_to_low_level_regions(section_savepath, deep_regions_lr, general_regions_lr, deep_regs_results):

    rspc = ReferenceSpaceCache(10, 'annotation/ccf_2017', manifest=MANIFEST_PATH)
    tree = rspc.get_structure_tree(structure_graph_id=1)
    acr_id = tree.get_id_acronym_map()
    id_acr = {v: k for k, v in acr_id.items()}

    # Sum high level values
    general_regions_cols = ['Animal', 'Rack', 'Slide', 'Section', 'type', 'Total'] + general_regions_lr
    general_regs_results = pd.DataFrame(columns=general_regions_cols)
    reportfile = open(os.path.join(section_savepath, "reportfile_low.txt"), 'w')
    for index, row in deep_regs_results.iterrows():
        count, existing = reset_count(general_regions_lr)
        if index > 0:
            if row['type'] != 'Density' and not pd.isnull(row['type']):
                for reg in deep_regions_lr:
                    if not pd.isnull(row[reg]):
                        suffix = reg[-3:]
                        region = reg[:-3]
                        if '_bg' in region:
                            if reg in count.keys():
                                count[acr] += float(row[reg])
                                existing[acr] = True
                        else:
                            parents_ids = tree.get_structures_by_acronym([region])[0]['structure_id_path'][2:]
                            parents_acr = [id_acr[id]+suffix for id in parents_ids]
                            for acr in parents_acr:
                                if acr in count.keys():
                                    count[acr] += float(row[reg])
                                    existing[acr] = True

            #total = np.sum(np.array(list(count.values())))
            total = '__'                
            blobs_color = row['type']
            
            if blobs_color == 'Red' or blobs_color == 'Green' or blobs_color == 'Coloc':
                total = row['Total']
                reportfile.write(f'\n \n \n {blobs_color} blobs:\t{str(int(row["Total"]))} \n ')
            
            data = {'Animal':row['Animal'], 'Rack': row['Rack'], 'Slide': row['Slide'], 'Section': row['Section'], 'type': row['type'], 'Total':total}

            obj = {}
            datafile = {}
            for acr in count.keys():
                if existing[acr]:
                    obj[acr] = count[acr]
                    if blobs_color == 'Red' or blobs_color == 'Green' or blobs_color == 'Coloc':
                        if int(count[acr]) != 0:
                            reportfile.write(f'\n {acr}:\t{str(int(count[acr]))}')
                else:
                    obj[acr] = np.nan
                    
            data.update(obj)

            general_regs_results = general_regs_results.append(data, ignore_index=True)
    reportfile.close()
    # Calculate density
    general_regions_cols = list(set(general_regs_results.columns) - set(['Animal', 'Rack', 'Slide', 'Section', 'type']))
    for index, row in general_regs_results.iterrows():
        if row['type'] == 'Density':
            for region in general_regions_cols:
                area = general_regs_results.iloc[index-1][region]
                green = general_regs_results.iloc[index-3][region]
                if not pd.isna(area) and not pd.isna(green):
                    if area != 0 and area != '__':
                        general_regs_results.loc[general_regs_results['type'] == 'Density', region] = green/int(area)
    return general_regs_results