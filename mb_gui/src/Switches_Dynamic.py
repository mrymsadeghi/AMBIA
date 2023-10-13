import os
import Switches_Static as st_switches

rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def set_rootpath(rootpath0):
    global rootpath
    rootpath = rootpath0
    return

def get_rootpath():
    return rootpath

if st_switches.atlas_type == 'Adult':
    from atlas_codes.Regions_n_colors_adult import Region_names
    region_names_for_alas_generation = []
    [region_names_for_alas_generation.append(item[:-3]) for item in Region_names if item[:-3] not in region_names_for_alas_generation]
elif st_switches.atlas_type == 'P56':
    from atlas_codes.Regions_n_colors_p56 import Region_names
    region_names_for_alas_generation = []
    [region_names_for_alas_generation.append(item[:-3]) for item in Region_names if item[:-3] not in region_names_for_alas_generation]

def set_ardent_reg_done_to_false():
    global ardent_reg_done
    ardent_reg_done = False
    return

def set_ardent_reg_done_to_true():
    global ardent_reg_done
    ardent_reg_done = True
    return

def get_status_of_ardent_reg_done(auto_registration_state):
    global ardent_reg_done
    if auto_registration_state:
        return ardent_reg_done
    else:
        return False

def set_reg_code(reg_code):
    global reg_code_status
    reg_code_status = reg_code
    return

def get_reg_code():
    return reg_code_status


def set_pred_atlas_sl_value(sl_value):
    global pred_sl_value
    pred_sl_value = sl_value
    return

def get_pred_sl_value():
    return pred_sl_value

def get_region_names_for_alas_generation():
    return region_names_for_alas_generation