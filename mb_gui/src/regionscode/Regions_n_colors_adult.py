
#from regions_per_sections_adult import regs_per_section, Regions_n_colors_List
from . import regions_per_sections_adult as rpsa
import numpy as np
import cv2
#import regions_per_sections_adult as rpsa
try :
    with open("./mb_gui/src/regionscode/regions_adult.txt","r") as file:# open("regionscode/regions_adult.txt","r") as file:#
        Region_names=file.read().split("\n")
        
except :
    with open("regionscode/regions_adult.txt","r") as file:#
        Region_names=file.read().split("\n")
def create_regs_n_colors_per_sec_list(atlasnum, tilted=False,labeled_atlas_filepath=None, 
    unlabeled_atlas_filepath=None,generated_3d_atlas_region_names=None,level_map_id_to_name=None,cm=None):

    if tilted:
        regs_n_colors_per_sec_List = [('root', 'not detected', '000000', (0, 0, 0))]
        generated_3d_atlas_region_names_unique=np.unique(generated_3d_atlas_region_names.flatten())
        for elem in generated_3d_atlas_region_names_unique:
            bb,gg,rr=cm[elem]
            regs_n_colors_per_sec_List.append(("empty_parent",
                                                level_map_id_to_name[elem],"hex_color",(bb,gg,rr)))
        Bgr_Color_List = []
        Rgb_Color_List = []
        for elem in regs_n_colors_per_sec_List:
            Rgb_Color_List.append(elem[-1])
            Bgr_Color_List.append((elem[-1][2],elem[-1][1],elem[-1][0]))
    else :
        regs_per_sec_List = rpsa.regs_per_section[int(atlasnum)] #list
        regs_n_colors_per_sec_List = [('root', 'not detected', '000000', (0, 0, 0))]
        for elem in rpsa.Regions_n_colors_List:
            if elem[-3] in regs_per_sec_List:
                regs_n_colors_per_sec_List.append(elem)
    
        Bgr_Color_List = []
        Rgb_Color_List = []
        for elem in regs_n_colors_per_sec_List:
            Rgb_Color_List.append(elem[-1])
            Bgr_Color_List.append((elem[-1][2],elem[-1][1],elem[-1][0]))

    return regs_n_colors_per_sec_List, Rgb_Color_List, Rgb_Color_List





#general_Region_names = ['FRP _R', 'FRP _L', 'MO _R', 'MO _L', 'SS _R', 'SS _L', 'GU _R', 'GU _L', 'VISC _R', 'VISC _L', 'AUD _R', 'AUD _L', 'VIS _R', 'VIS _L', 'ACA _R', 'ACA _L', 'PL _R', 'PL _L', 'ILA _R', 'ILA _L', 'ORB _R', 'ORB _L', 'AI _R', 'AI _L', 'RSP _R', 'RSP _L', 'PTLp _R', 'PTLp _L', 'TEa _R', 'TEa _L', 'PERI _R', 'PERI _L', 'ECT _R', 'ECT _L', 'MOB _R', 'MOB _L', 'AOB _R', 'AOB _L', 'AON _R', 'AON _L', 'TT _R', 'TT _L', 'DP _R', 'DP _L', 'PIR _R', 'PIR _L', 'NLOT _R', 'NLOT _L', 'COA _R', 'COA _L', 'PAA _R', 'PAA _L', 'TR _R', 'TR _L', 'HIP _R', 'HIP _L', 'RHP _R', 'RHP _L', 'CLA _R', 'CLA _L', 'EP _R', 'EP _L', 'LA _R', 'LA _L', 'BLA _R', 'BLA _L', 'BMA _R', 'BMA _L', 'PA _R', 'PA _L', 'STRd _R', 'STRd _L', 'STRv _R', 'STRv _L', 'LSX _R', 'LSX _L', 'sAMY _R', 'sAMY _L', 'PALd _R', 'PALd _L', 'PALv _R', 'PALv _L', 'PALm _R', 'PALm _L', 'PALc _R', 'PALc _L', 'VENT _R', 'VENT _L', 'SPF _R', 'SPF _L', 'SPA _R', 'SPA _L', 'PP _R', 'PP _L', 'GENd _R', 'GENd _L', 'LAT _R', 'LAT _L', 'ATN _R', 'ATN _L', 'MED _R', 'MED _L', 'MTN _R', 'MTN _L', 'ILM _R', 'ILM _L', 'RT _R', 'RT _L', 'GENv _R', 'GENv _L', 'EPI _R', 'EPI _L', 'PVZ _R', 'PVZ _L', 'PVR _R', 'PVR _L', 'MEZ _R', 'MEZ _L', 'LZ _R', 'LZ _L', 'ME _R', 'ME _L', 'SCs _R', 'SCs _L', 'IC _R', 'IC _L', 'NB _R', 'NB _L', 'SAG _R', 'SAG _L', 'PBG _R', 'PBG _L', 'MEV _R', 'MEV _L', 'SNr _R', 'SNr _L', 'VTA _R', 'VTA _L', 'RR _R', 'RR _L', 'MRN _R', 'MRN _L', 'SCm _R', 'SCm _L', 'PAG _R', 'PAG _L', 'PRT _R', 'PRT _L', 'CUN _R', 'CUN _L', 'RN _R', 'RN _L', 'III _R', 'III _L', 'EW _R', 'EW _L', 'IV _R', 'IV _L', 'VTN _R', 'VTN _L', 'AT _R', 'AT _L', 'LT _R', 'LT _L', 'SNc _R', 'SNc _L', 'PPN _R', 'PPN _L', 'RAmb _R', 'RAmb _L', 'P-sen _R', 'P-sen _L', 'P-mot _R', 'P-mot _L', 'P-sat _R', 'P-sat _L', 'MY-sen _R', 'MY-sen _L', 'MY-mot _R', 'MY-mot _L', 'MY-sat _R', 'MY-sat _L', 'LING _R', 'LING _L', 'CENT _R', 'CENT _L', 'CUL _R', 'CUL _L', 'DEC _R', 'DEC _L', 'FOTU _R', 'FOTU _L', 'PYR _R', 'PYR _L', 'UVU _R', 'UVU _L', 'NOD _R', 'NOD _L', 'SIM _R', 'SIM _L', 'AN _R', 'AN _L', 'PRM _R', 'PRM _L', 'COPY _R', 'COPY _L', 'PFL _R', 'PFL _L', 'FL _R', 'FL _L', 'FN _R', 'FN _L', 'IP _R', 'IP _L', 'DN _R', 'DN _L', 'OLF _R', 'OLF _L', 'CTX _R', 'CTX _L', 'HY _R', 'HY _L', 'TH _R', 'TH _L', 'MB _R', 'MB _L', 'P _R', 'P _L', 'MY _R', 'MY _L', 'CB _R', 'CB _L', 'VS _R', 'VS _L', 'fiber tracts _R', 'fiber tracts _L', 'not detected _R', 'not detected _L', 'CTX_bg _L', 'CTX_bg _R', 'CTXsp_bg _L', 'CTXsp_bg _R', 'STR_bg _L', 'STR_bg _R', 'PAL_bg _L', 'PAL_bg _R', 'TH_bg _L', 'TH_bg _R', 'HY_bg _L', 'HY_bg _R', 'MB_bg _L', 'MB_bg _R', 'P_bg _L', 'P_bg _R', 'MY_bg _L', 'MY_bg _R', 'CB_bg _L', 'CB_bg _R', 'FN_bg _L', 'FN_bg _R', 'IP_bg _L', 'IP_bg _R', 'DN_bg _L', 'DN_bg _R', 'VeCB_bg _L', 'VeCB_bg _R', 'fiber tracts_bg _L', 'fiber tracts_bg _R', 'VS_bg _L', 'VS_bg _R']
general_Region_names =['FRP_R', 'FRP_L', 'MO_R', 'MO_L', 'SS_R', 'SS_L', 'GU_R', 'GU_L', 'VISC_R', 'VISC_L', 'AUD_R', 'AUD_L', 'VIS_R', 'VIS_L', 'ACA_R', 'ACA_L', 'PL_R', 'PL_L', 'ILA_R', 'ILA_L', 'ORB_R', 'ORB_L', 'AI_R', 'AI_L', 'RSP_R', 'RSP_L', 'PTLp_R', 'PTLp_L', 'TEa_R', 'TEa_L', 'PERI_R', 'PERI_L', 'ECT_R', 'ECT_L', 'MOB_R', 'MOB_L', 'AOB_R', 'AOB_L', 'AON_R', 'AON_L', 'TT_R', 'TT_L', 'DP_R', 'DP_L', 'PIR_R', 'PIR_L', 'NLOT_R', 'NLOT_L', 'COA_R', 'COA_L', 'PAA_R', 'PAA_L', 'TR_R', 'TR_L', 'HIP_R', 'HIP_L', 'RHP_R', 'RHP_L', 'CLA_R', 'CLA_L', 'EP_R', 'EP_L', 'LA_R', 'LA_L', 'BLA_R', 'BLA_L', 'BMA_R', 'BMA_L', 'PA_R', 'PA_L', 'STRd_R', 'STRd_L', 'STRv_R', 'STRv_L', 'LSX_R', 'LSX_L', 'sAMY_R', 'sAMY_L', 'PALd_R', 'PALd_L', 'PALv_R', 'PALv_L', 'PALm_R', 'PALm_L', 'PALc_R', 'PALc_L', 'VENT_R', 'VENT_L', 'SPF_R', 'SPF_L', 'SPA_R', 'SPA_L', 'PP_R', 'PP_L', 'GENd_R', 'GENd_L', 'LAT_R', 'LAT_L', 'ATN_R', 'ATN_L', 'MED_R', 'MED_L', 'MTN_R', 'MTN_L', 'ILM_R', 'ILM_L', 'RT_R', 'RT_L', 'GENv_R', 'GENv_L', 'EPI_R', 'EPI_L', 'PVZ_R', 'PVZ_L', 'PVR_R', 'PVR_L', 'MEZ_R', 'MEZ_L', 'LZ_R', 'LZ_L', 'ME_R', 'ME_L', 'SCs_R', 'SCs_L', 'IC_R', 'IC_L', 'NB_R', 'NB_L', 'SAG_R', 'SAG_L', 'PBG_R', 'PBG_L', 'MEV_R', 'MEV_L', 'SNr_R', 'SNr_L', 'VTA_R', 'VTA_L', 'RR_R', 'RR_L', 'MRN_R', 'MRN_L', 'SCm_R', 'SCm_L', 'PAG_R', 'PAG_L', 'PRT_R', 'PRT_L', 'CUN_R', 'CUN_L', 'RN_R', 'RN_L', 'III_R', 'III_L', 'EW_R', 'EW_L', 'IV_R', 'IV_L', 'VTN_R', 'VTN_L', 'AT_R', 'AT_L', 'LT_R', 'LT_L', 'SNc_R', 'SNc_L', 'PPN_R', 'PPN_L', 'RAmb_R', 'RAmb_L', 'P-sen_R', 'P-sen_L', 'P-mot_R', 'P-mot_L', 'P-sat_R', 'P-sat_L', 'MY-sen_R', 'MY-sen_L', 'MY-mot_R', 'MY-mot_L', 'MY-sat_R', 'MY-sat_L', 'LING_R', 'LING_L', 'CENT_R', 'CENT_L', 'CUL_R', 'CUL_L', 'DEC_R', 'DEC_L', 'FOTU_R', 'FOTU_L', 'PYR_R', 'PYR_L', 'UVU_R', 'UVU_L', 'NOD_R', 'NOD_L', 'SIM_R', 'SIM_L', 'AN_R', 'AN_L', 'PRM_R', 'PRM_L', 'COPY_R', 'COPY_L', 'PFL_R', 'PFL_L', 'FL_R', 'FL_L', 'FN_R', 'FN_L', 'IP_R', 'IP_L', 'DN_R', 'DN_L', 'OLF_R', 'OLF_L', 'CTX_R', 'CTX_L', 'HY_R', 'HY_L', 'TH_R', 'TH_L', 'MB_R', 'MB_L', 'P_R', 'P_L', 'MY_R', 'MY_L', 'CB_R', 'CB_L', 'VS_R', 'VS_L', ' tracts_R', 'fiber tracts_L', 'not detected_R', 'not detected_L', 'CTX_bg_L', 'CTX_bg_R', 'CTXsp_bg_L', 'CTXsp_bg_R', 'STR_bg_L', 'STR_bg_R', 'PAL_bg_L', 'PAL_bg_R', 'TH_bg_L', 'TH_bg_R', 'HY_bg_L', 'HY_bg_R', 'MB_bg_L', 'MB_bg_R', 'P_bg_L', 'P_bg_R', 'MY_bg_L', 'MY_bg_R', 'CB_bg_L', 'CB_bg_R', 'FN_bg_L', 'FN_bg_R', 'IP_bg_L', 'IP_bg_R', 'DN_bg_L', 'DN_bg_R', 'VeCB_bg_L', 'VeCB_bg_R', 'fiber tracts_bg_L', 'fiber tracts_bg_R', 'VS_bg_L', 'VS_bg_R']
