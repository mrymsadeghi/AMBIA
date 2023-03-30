import os
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from easysettings import EasySettings
from utils.img_processing import rotate_img, equalize_img
from utils.reading_czi import czi_section_img, czi_channel_regulator, czi_preview,histogram_equalization, get_channel_info
from atlas_codes.regions_per_sections_p56 import regs_per_section
import Switches_Static as st_switches
from utils.allen_functions import high_to_low_level_regions
import AMBIA_M1_CellDetection as m1 
MBGUI_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
OPENSLIDE_PATH = os.path.join(MBGUI_PATH, "openslide_dlls", "bin")
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide



settings = EasySettings("myconfigfile.conf")
if st_switches.atlas_type == "Adult":
    from atlas_codes.Regions_n_colors_adult import Region_names, general_Region_names, create_regs_n_colors_per_sec_list
elif st_switches.atlas_type == "P56":
    from atlas_codes.Regions_n_colors_p56 import Region_names, general_Region_names, create_regs_n_colors_per_sec_list




num_rows = st_switches.num_rows

#Global variables
global alevel, blevel, mlevel, maskthresh
global dfm0, dfma, dfmb, dfba
global MARGIN
MARGIN = st_switches.MARGIN

#Global Functions
def coords_to_colorindex(pointcolor , section):
    if pointcolor in section.Bgr_Color_list:
        colorindex = section.Bgr_Color_list.index(pointcolor)
    else:
        colorindex = 0
    return colorindex

def get_region_color(regmargin, section_roi):
    sectioncolors = []
    for i in range (0, regmargin*2):
        for j in range(0,regmargin*2):
            b,g,r = section_roi[i,j]
            sectioncolors.append((r,g,b))
    pointcolor2 = max(set(sectioncolors), key = sectioncolors.count)
    return pointcolor2

def save_to_saved_data_pickle(item, keytext, section):
    section.saved_data_pickle[keytext] = item

def save_the_pickle(section):
    section.save_to_pkl("precious_saved_data.pkl", section.saved_data_pickle)
    return


# Slide Class
class Slide:
    def __init__(self , section=None , prepath=None ,  slidename=None , slidepath = None , Experiment_num = None , rack_num = None , slide_num = None , slideformat = None , savepath = None , brnum = None , Report_df = None , channel_types = None , slideimgpath = None , tissuemask_fullpath = None ,brainboundcoords = None ,):
        self.section = section
        self.prepath = prepath
        self.slidename = slidename
        self.slidepath  = slidepath
        self.Experiment_num = Experiment_num
        self.rack_num = rack_num
        self.slide_num = slide_num
        self.slideformat = slideformat
        self.savepath = savepath
        self.brnum = brnum
        self.Report_df = Report_df
        self.channel_types = channel_types
        self.slideimgpath = slideimgpath
        self.tissuemask_fullpath = tissuemask_fullpath
        self.brainboundcoords = brainboundcoords

    def funcReadSlide(self , slide_fullpath, prepath0):
        global alevel, blevel, mlevel, maskthresh
        global dfm0, dfma, dfmb, dfba
         
        self.prepath = prepath0
        
        try:
            self.slidename = os.path.split(slide_fullpath)[-1].split(".")[0]
            self.slidepath = slide_fullpath
            self.Experiment_num = self.slidename.split("_")[0][1:]
            self.rack_num = self.slidename.split("_")[1][1:]
            self.slide_num = self.slidename.split("_")[2][1:]
        except IndexError:
            print("Format of the slide name is not correct")
            return "em1"
        self.slideformat = os.path.split(slide_fullpath)[-1].split(".")[1].lower()
        self.savepath = os.path.join(self.prepath, self.slidename)
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        #Reportxls_Initialize(self.slidename, self.savepath, self.Experiment_num, Animal_num, self.slide_num)
        #### Report Dataframe
        template_path =  os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "models", "templates")
        report_xls_path = os.path.join(self.savepath, f'Report_{self.slidename}.xlsx')
        if os.path.exists(report_xls_path):
            self.Report_df = pd.read_excel(report_xls_path)
            print("xlsx exists")
        else:
            if self.slideformat == "mrxs":
                #columns = ['id', 'Animal', 'Rack', 'Slide', 'Section', 'type', 'Total'] + Region_names
                self.Report_df = pd.read_excel(os.path.join(template_path, 'ambia_template_mrx.xlsx'))
            elif self.slideformat == "czi":
                #columns=['id','Experiment', 'Animal', 'Slide', 'Section', 'type', 'Total'] + Region_names
                self.Report_df = pd.read_excel(os.path.join(template_path, 'ambia_template_czi.xlsx'))
        print('Slide path: ', slide_fullpath)
        if self.slideformat == "mrxs":
            print("It is mirax format")
            mlevel = st_switches.mrx_mlevel  # mask level
            blevel = st_switches.mrx_blevel  # Blob detection level
            alevel = st_switches.mrx_alevel  # Atlas mapping level
            Slide =  openslide.OpenSlide(slide_fullpath)
            Dims = Slide.level_dimensions
            slideimage = Slide.read_region((0, 0), mlevel, (Dims[mlevel][0], Dims[mlevel][1]))
            slideimage = cv.cvtColor(np.array(slideimage), cv.COLOR_RGB2BGR)
            slideimage_equalized = equalize_img(slideimage)
            slideimage = rotate_img(slideimage_equalized)
            self.slideimgpath = os.path.join(self.savepath, "mlevel.jpg")
            cv.imwrite(self.slideimgpath, slideimage)
            self.num_channels = st_switches.mrx_num_channels
            self.channel_types = ["B", "G", "R"]
            maskthresh = st_switches.mrx_maskthresh
        elif self.slideformat == "czi":
            print("It is czi format")
            mlevel = st_switches.czi_mlevel  # mask level
            blevel = st_switches.czi_blevel  # Blob detection level
            alevel = st_switches.czi_alevel  # Atlas mapping level
            self.slideimgpath = os.path.join(self.savepath, "mlevel.jpg")
            self.num_channels, self.channel_types = get_channel_info(slide_fullpath)
            czi_preview(slide_fullpath, self.slideimgpath, mlevel)
            maskthresh = st_switches.czi_maskthresh
        else:
            print("Unsupported slide format")

        dfm0 = 2 ** mlevel  # Convert factor from mask level to level zero
        dfma = 2 ** (mlevel - alevel)
        dfmb = 2 ** (mlevel - blevel)
        dfba = 2 ** (alevel - blevel)
        return self.slideimgpath

    def funcSectionDetection(self):
         
        """
        Detects individual sections
        Outputs bounding box corrds for each section in brainboundcoords
        self.brainboundcoords = list of [x, y, w, h]
        """
        self.savepath = os.path.join(self.prepath, self.slidename)
        self.brainboundcoords = np.array([[0, 0, 0, 0]])  #[x, y, w, h]
        slideimg = cv.imread(os.path.join(self.savepath, "mlevel.jpg"))
        imgray = cv.cvtColor(slideimg, cv.COLOR_BGR2GRAY)
        img_gr_mblur = cv.medianBlur(imgray, 9)
        # In mask level specified above
        _, slidemask = cv.threshold(img_gr_mblur, maskthresh, 255, cv.THRESH_BINARY)  # mlevel
        cv.imwrite(os.path.join(self.savepath, "tissue_mask.jpg"), slidemask)
        cv.imwrite(os.path.join(self.savepath, "imgray.jpg"), imgray)
        contours, _ = cv.findContours(slidemask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # mlevel

        ### removing small contours
        contours=list(contours)
        threshold_area = 10000
        for i, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            if area < threshold_area:
                #small_contours.append(i)
                contours.pop(i)
        # Finding the rearranging order
        br_min_coords = []
        br_min_coords_by5 = []
        for i, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            mins = np.amin(cnt, axis=0)
            br_min_coords.append((x, y))
            br_min_coords_by5.append(x + y * 5)

        br_min_coords_by5_sorted = sorted(br_min_coords_by5)
        rearrange_guide = [(i, j) for i in range(len(br_min_coords_by5)) for j in range(len(br_min_coords_by5_sorted)) if
                        br_min_coords_by5[i] == br_min_coords_by5_sorted[j]]
        rearrange_guide_sorted = sorted(rearrange_guide, key=lambda tup: tup[1])
        brnumtemp = 1
        for indextuple in rearrange_guide_sorted:
            indexnew = indextuple[0]
            cnt = contours[indexnew]
            area = cv.contourArea(cnt)
            # get the bounding rect
            x, y, w, h = cv.boundingRect(cnt)  # mlevel        
            self.brainboundcoords  = np.append(self.brainboundcoords , [[x, y, w, h]], axis=0)
            brnumtemp += 1  
        ########### Report
        num_sections = len(self.brainboundcoords )
        
        if self.slideformat == "mrxs":
            dict_base = {'Animal': self.Experiment_num, 'Rack': self.rack_num, 'Slide': self.slide_num}
            self.tissuemask_fullpath = os.path.join(self.savepath, "tissue_mask.jpg")
        elif self.slideformat == "czi":
            dict_base = {'Experiment': self.Experiment_num, 'Animal': self.rack_num, 'Slide': self.slide_num}
            self.tissuemask_fullpath = os.path.join(self.savepath, "imgray.jpg")

        if self.Report_df.shape[0] < 9:
            for i in range(8, num_sections*num_rows+8):
                dict_base['id']=i+1
                self.Report_df = self.Report_df.append(dict_base, ignore_index=True)

        self.brainboundcoords  = self.brainboundcoords [1:, :]
        self.section = Section(self.slideformat, self.slidepath , self.prepath , self.slidename ,  self.num_channels , self.Experiment_num   , self.Report_df , self.savepath , self.rack_num , self.slide_num)
        return self.brainboundcoords , self.tissuemask_fullpath

# Section Class
class Section:
    def __init__(self , slideformat , slidepath , prepath , slidename ,  num_channels , Experiment_num   , Report_df , savepath  , rack_num , slide_num , saved_data_pickle = {} , brnum = None , blobs_log_g = None , blobs_log_r = None , section_savepath = None , Bgr_Color_list = None , Report_subdf = None ):
        self.slideformat = slideformat
        self.slidepath = slidepath
        self.prepath = prepath
        self.slidename = slidename
        self.num_channels = num_channels
        self.Experiment_num = Experiment_num
        self.Report_df = Report_df
        self.savepath = savepath
        self.rack_num = rack_num
        self.slide_num = slide_num
        self.saved_data_pickle = saved_data_pickle
        self.brnum = brnum
        self.blobs_log_g =blobs_log_g
        self.blobs_log_r = blobs_log_r
        self.section_savepath = section_savepath
        self.Bgr_Color_list = Bgr_Color_list
        self.Report_subdf = Report_subdf

    def get_section_images(self , brnum0, brainboundcoords):
        global alevel, blevel, mlevel, maskthresh
        global dfm0, dfma, dfmb, dfba
        self.section_savepath = os.path.join(self.savepath, f"S{brnum0}")
        if not os.path.exists(self.section_savepath):
            os.makedirs(self.section_savepath)
        if self.slideformat == "mrxs":
            x, y, w, h = brainboundcoords[brnum0-1]  # mlevel
            [xb, yb, wb, hb] = [x * dfmb, y * dfmb, w * dfmb, h * dfmb]
            [xa, ya, wa, ha] = [x * dfma, y * dfma, w * dfma, h * dfma]
            Slidefile = openslide.OpenSlide(self.slidepath)
            Dims = Slidefile.level_dimensions
            brainimg = Slidefile.read_region((y * dfm0, Dims[0][1] - ((x + w) * dfm0)), blevel, (hb, wb)).convert("RGB")
            brainimg2 = np.array(brainimg)
            section_blevel = rotate_img(brainimg2)
            if st_switches.color_switch_on:
                section_blevel2 = section_blevel.copy()
                section_blevel2[:,:,1]= section_blevel[:,:,2]
                section_blevel2[:,:,2]= section_blevel[:,:,1]
                section_blevel = section_blevel2
            section_blevel_eq = equalize_img(section_blevel)
            braina = Slidefile.read_region((y * dfm0, Dims[0][1] - ((x + w) * dfm0)), alevel, (ha, wa)).convert("RGB")
            braina_dark = np.array(braina)
            braina_rot = rotate_img(braina_dark)
            section_alevel = cv.copyMakeBorder(braina_rot, MARGIN, MARGIN, MARGIN, MARGIN, cv.BORDER_CONSTANT,
                                            value=(0, 0, 0))
            if st_switches.color_switch_on:
                section_alevel2 = section_alevel.copy()
                section_alevel2[:,:,1]= section_alevel[:,:,2]
                section_alevel2[:,:,2]= section_alevel[:,:,1]
                section_alevel = section_alevel2
            section_alevel_eq = equalize_img(section_alevel)
            blevel_b, blevel_g, blevel_r = cv.split(section_blevel)
            cv.imwrite(os.path.join(self.section_savepath,"alevel.png"), section_alevel)
            cv.imwrite(os.path.join(self.section_savepath,"blevel.png"), section_blevel)
            cv.imwrite(os.path.join(self.section_savepath,"alevel_eq.png"), section_alevel_eq)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_eq.png"), section_blevel_eq)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_b.png"), blevel_b)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_g.png"), blevel_g)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_r.png"), blevel_r)
        
        elif self.slideformat == "czi":
            num_sections = len(brainboundcoords)
            section_alevel = czi_section_img(self.slidepath, brnum0, num_sections, alevel, [], rect=None)
            section_blevel = czi_section_img(self.slidepath, brnum0, num_sections, blevel, [], rect=None)
            section_alevel = czi_channel_regulator(section_alevel)
            section_blevel = czi_channel_regulator(section_blevel)
            section_alevel = cv.copyMakeBorder(section_alevel, MARGIN, MARGIN, MARGIN, MARGIN, cv.BORDER_CONSTANT,
                                            value=(0, 0, 0))
            section_alevel_eq = histogram_equalization(section_alevel)
            section_blevel_eq = histogram_equalization(section_blevel)

            cv.imwrite(os.path.join(self.section_savepath,"alevel.png"), section_alevel)
            cv.imwrite(os.path.join(self.section_savepath,"blevel.png"), section_blevel)
            cv.imwrite(os.path.join(self.section_savepath,"alevel_eq.png"), section_alevel_eq)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_eq.png"), section_blevel_eq)
            for channel in range(self.num_channels):
                # channel_name = self.channel_types[channel]
                blevel_channel = czi_section_img(self.slidepath, brnum0, num_sections, blevel, [channel], rect=None)
                cv.imwrite(os.path.join(self.section_savepath, f"blevel_{channel}.png"), blevel_channel)
            if self.num_channels ==2:
                a = np.zeros(shape=(blevel_channel.shape[0],blevel_channel.shape[1]))
                cv.imwrite(os.path.join(self.section_savepath, f"blevel_2.png"), blevel_channel)
        
        blob_detection_file_name = os.path.join(self.section_savepath,"blevel_eq.png")
        tissue_lm_detection_filename = os.path.join(self.section_savepath,"alevel_eq.png")
        return blob_detection_file_name, tissue_lm_detection_filename

    def section_flip_operation(self):

         
        section_alevel = cv.flip(cv.imread(os.path.join(self.section_savepath,"alevel.png")), 1)
        section_blevel = cv.flip(cv.imread(os.path.join(self.section_savepath,"blevel.png")), 1)
        section_alevel_eq = cv.flip(cv.imread(os.path.join(self.section_savepath,"alevel_eq.png")), 1)
        section_blevel_eq = cv.flip(cv.imread(os.path.join(self.section_savepath,"blevel_eq.png")), 1)
        if self.slideformat == "mrxs":
            blevel_b, blevel_g, blevel_r = cv.split(section_blevel)

            cv.imwrite(os.path.join(self.section_savepath,"blevel_b.png"), blevel_b)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_g.png"), blevel_g)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_r.png"), blevel_r)
        elif self.slideformat == "czi":
            blevel_ch0 = cv.flip(cv.imread(os.path.join(self.section_savepath,"blevel_0.png")), 1)
            blevel_ch1 = cv.flip(cv.imread(os.path.join(self.section_savepath,"blevel_1.png")), 1)
            blevel_ch2 = cv.flip(cv.imread(os.path.join(self.section_savepath,"blevel_2.png")), 1)

            cv.imwrite(os.path.join(self.section_savepath,"blevel_ch0.png"), blevel_ch0)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_ch1.png"), blevel_ch1)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_ch2.png"), blevel_ch2)

        cv.imwrite(os.path.join(self.section_savepath,"alevel.png"), section_alevel)
        cv.imwrite(os.path.join(self.section_savepath,"blevel.png"), section_blevel)
        cv.imwrite(os.path.join(self.section_savepath,"alevel_eq.png"), section_alevel_eq)
        cv.imwrite(os.path.join(self.section_savepath,"blevel_eq.png"), section_blevel_eq)
        blob_detection_file_name = os.path.join(self.section_savepath,"blevel_eq.png")
        tissue_lm_detection_filename = os.path.join(self.section_savepath,"alevel_eq.png")
        return blob_detection_file_name, tissue_lm_detection_filename

    def funcBlobDetection(self , brnum, blobs_parameters):
         
        """ Returns self.blobs_log_r, blobs_log_g, colocalized_blobs:: list of blob coords (r,c) before 
        adding the blobs added/removed manually by user
        Saves blobs_locs_r, blobs_locs_g, blob_locs_co as numpy.array
        this also include blob coords (r,c) before adding the blobs added/removed manually by user
        Saves  r_params_for_save, g_params_for_save as npy
        """



        tempMARGIN = 50  # temporary margin just to avoid the borders when applying thresh, adding the margin is reversed in the parameter brain_mask_eroded

        brainimgtemp = cv.imread(os.path.join(self.section_savepath, 'blevel_eq.png'))
        brainimgtemp = cv.copyMakeBorder(brainimgtemp, tempMARGIN, tempMARGIN, tempMARGIN, tempMARGIN, cv.BORDER_CONSTANT, value=(0, 0, 0))
        brainimgtemp_gray = cv.cvtColor(brainimgtemp, cv.COLOR_BGR2GRAY)
        _, brain_mask = cv.threshold(brainimgtemp_gray, 5, 255, cv.THRESH_BINARY)
        cv.imwrite(os.path.join(self.section_savepath, 'brain_mask.jpg'), brain_mask[tempMARGIN:-tempMARGIN, tempMARGIN:-tempMARGIN])

        if self.slideformat == "mrxs":
            kernel1 = np.ones((11,11), np.uint8)
            kernel2 = np.ones((27, 27), np.uint8)
            img_channel_r = cv.imread(os.path.join(self.section_savepath, 'blevel_r.png'), 0)
            img_channel_g = cv.imread(os.path.join(self.section_savepath, 'blevel_g.png'), 0)

        elif self.slideformat == "czi":
            kernel1 = np.ones((11,11), np.uint8)
            kernel2 = np.ones((11, 11), np.uint8)
            img_channel_r = cv.imread(os.path.join(self.section_savepath, 'blevel_2.png'), 0)
            img_channel_g = cv.imread(os.path.join(self.section_savepath, 'blevel_1.png'), 0)

        closing = cv.morphologyEx(brain_mask, cv.MORPH_CLOSE, kernel2)
        #cv.imwrite(os.path.join(self.section_savepath, 'brain_mask_closed.jpg'), closing)
        brain_mask_eroded_uncut = cv.erode(closing, kernel2, iterations=3)
        #cv.imwrite(os.path.join(self.section_savepath, 'brain_mask_eroded.jpg'), brain_mask_eroded_uncut)

        brain_mask_eroded = brain_mask_eroded_uncut[tempMARGIN:-tempMARGIN, tempMARGIN:-tempMARGIN]
        cv.imwrite(os.path.join(self.section_savepath, 'brain_mask_eroded_cut.jpg'), brain_mask_eroded)

        img_channel_g = cv.medianBlur(img_channel_g, 3)
        img_not_r = cv.medianBlur(img_channel_r, 3)

        ### Parameters
        red_blob_type = blobs_parameters["red_blob_type"]
        green_blob_type = blobs_parameters["green_blob_type"]
        green_blobs_thresh = blobs_parameters["green_blob_thresh"]
        number_of_blobs_g = 0
        number_of_blobs_r = 0
        blobs_parameters_dict_to_save = {}
        ######### Red blobs detection

        if red_blob_type == "Rabies":
            minsize = blobs_parameters['red_blob_min_size']
            red_blobs_thresh = blobs_parameters["red_blob_thresh"]
            self.blobs_log_r = m1.rabies_detection(img_channel_r, red_blobs_thresh, minsize, brain_mask_eroded)
            #r_params_for_save = np.array([minsize,red_blobs_thresh])
            blobs_parameters_dict_to_save['red'] = [minsize, red_blobs_thresh]
            #np.save(os.path.join(self.section_savepath, 'blobparams_r.npy'), r_params_for_save)

        if red_blob_type == "MoG":
            min_corr = blobs_parameters['red_blob_correlation']
            stride = blobs_parameters['red_blob_stride']
            self.blobs_log_r = m1.MoG_detection(img_channel_r, min_corr, stride, brain_mask_eroded)

        elif red_blob_type == "cFos":
            minsigma = blobs_parameters['red_blob_min_sigma']
            maxsigma = blobs_parameters['red_blob_max_sigma']
            numsigma = blobs_parameters['red_blob_num_sigma']
            red_blobs_thresh = blobs_parameters['red_blob_thresh2'] /100
            #self.blobs_log_r = cfos_detection(img_channel_r, minsigma, maxsigma, numsigma, red_blobs_thresh, brain_mask_eroded)
            self.blobs_log_r = m1.pool_cell_detetcion(img_channel_r, brain_mask_eroded, minsigma, maxsigma, numsigma, red_blobs_thresh)
            blobs_parameters_dict_to_save['red'] = [minsigma, maxsigma, numsigma, red_blobs_thresh]
        blob_locs_r = np.array(self.blobs_log_r)
        number_of_blobs_r = len(self.blobs_log_r)
        ####### Green blobs detection
        if green_blob_type == "Rabies":
            minsize = blobs_parameters['green_blob_min_size']
            green_blobs_thresh = blobs_parameters['green_blob_thresh']
            self.blobs_log_g = m1.rabies_detection(img_channel_g, green_blobs_thresh, minsize, brain_mask_eroded)
            #g_params_for_save = np.array()
            blobs_parameters_dict_to_save['green'] = [minsize,green_blobs_thresh]

        elif green_blob_type == "MoG":
            min_corr = blobs_parameters['green_blob_correlation']
            stride = blobs_parameters['green_blob_stride']
            self.blobs_log_g = m1.MoG_detection(img_channel_g, min_corr, stride, brain_mask_eroded)

        elif green_blob_type == "cFos":
            minsigma = blobs_parameters['green_blob_min_sigma']
            maxsigma = blobs_parameters['green_blob_max_sigma']
            numsigma = blobs_parameters['green_blob_num_sigma']
            green_blobs_thresh = blobs_parameters['green_blob_thresh2']/100
            
            self.blobs_log_g = m1.pool_cell_detetcion(img_channel_g, brain_mask_eroded, minsigma, maxsigma, numsigma, green_blobs_thresh)


            blobs_parameters_dict_to_save['green'] = [minsigma, maxsigma, numsigma, green_blobs_thresh]
            #np.save(os.path.join(self.section_savepath, 'blobparams_g.npy'), g_params_for_save)
        matchcount, blob_locs_co = m1.calculate_colocalized_blobs(self.blobs_log_r, self.blobs_log_g)
        blob_locs_g = np.array(self.blobs_log_g)
        number_of_blobs_g = len(self.blobs_log_g)
        #save_to_pkl("blobs_parameters", blobs_parameters)
        self.saved_data_pickle['blobs_parameters'] = blobs_parameters_dict_to_save
        ####### colocalized
        screenimg_path = os.path.join(self.section_savepath, 'blevel_eq.png')
        #np.save(os.path.join(self.section_savepath, "bloblocs_g_auto.npy"), blob_locs_g)
        #np.save(os.path.join(self.section_savepath, "bloblocs_r_auto.npy"), blob_locs_r)
        #np.save(os.path.join(self.section_savepath, "bloblocs_co_auto.npy"), blob_locs_co)
        return number_of_blobs_r, number_of_blobs_g, matchcount, screenimg_path, self.blobs_log_r, self.blobs_log_g, blob_locs_co

    def save_to_pkl(self , filename, data):
         
        b_file = open(os.path.join(self.section_savepath, filename), "wb")
        pickle.dump(data, b_file)
        b_file.close()

    def funcLandmarkDetection(self , imgpath, midheight):
        # midh = 200 for tissue
        # midh = 400 for atlas

        LandmarksT = []
        return LandmarksT

    def calculate_fp_fn_blobs(self , red_blobs_modified, green_blobs_modified):
        blobs_fp_fn = {}
        red_blobs_modified2 = [(sub[1], sub[0]) for sub in red_blobs_modified] 
        green_blobs_modified2 = [(sub[1], sub[0]) for sub in green_blobs_modified] 
        blobs_fp_fn['red_fn'] = [item for item in red_blobs_modified2 if item not in self.blobs_log_r]   #Added_red_points
        blobs_fp_fn['red_fp'] = [item for item in self.blobs_log_r if item not in red_blobs_modified2]  #Removed red points
        blobs_fp_fn['green_fn'] = [item for item in green_blobs_modified2 if item not in self.blobs_log_g]   #Added_green_points
        blobs_fp_fn['green_fp'] = [item for item in self.blobs_log_g if item not in green_blobs_modified2]  #Removed green points
        reportfile = open(os.path.join(self.section_savepath, "reportfile_fpfn.txt"), 'w')
        reportfile.write(f'\n Red FP: {len(blobs_fp_fn["red_fp"])} and FN: {len(blobs_fp_fn["red_fn"])}')
        reportfile.write(f'\n Green FP: {len(blobs_fp_fn["green_fp"])} and FN: {len(blobs_fp_fn["green_fn"])}')
        reportfile.close()
        #blob_locs_r = np.array(red_blobs_modified2)
        #blob_locs_g = np.array(green_blobs_modified2)
        self.saved_data_pickle['blobs_fp_fn'] = blobs_fp_fn
        #save_to_pkl("blobs_coords_fp_fn.pkl", blobs_fp_fn)
        return


    def funcAnalysis(self, atlasnum, brnum, atlas_prepath, red_blobs_modified, green_blobs_modified, colocalized_blobs_coords) :

         
        """ Inputs red/green_blobs_modified as a list of blob coords (c, r)
        these include detected blob coords after user modification
        red/green_blobs_modified coords are in blevel
        Intermediate variable blob_locs_r/g np array  [r,c]
        """
        self.brnum = brnum


        regmargin = 5  #for color averaging

        if self.slideformat == "mrxs":
            dict_base = {'Animal': self.Experiment_num, 'Rack': self.rack_num, 'Slide': self.slide_num, 'Section': self.brnum}
            self.Report_subdf = pd.DataFrame(columns=['id', 'Animal', 'Rack', 'Slide', 'Section', 'type', 'Total'] + Region_names)
        elif self.slideformat == "czi":    
            dict_base = {'Experiment': self.Experiment_num, 'Animal': self.rack_num, 'Slide': self.slide_num, 'Section': self.brnum}
            self.Report_subdf = pd.DataFrame(columns=['id', 'Experiment', 'Animal', 'Slide', 'Section', 'type', 'Total'] + Region_names)

        Regions_n_colors_list, self.Bgr_Color_list, Rgb_Color_list =  create_regs_n_colors_per_sec_list(atlasnum)

        self.savepath = os.path.join(self.prepath, self.slidename)
        labeled_atlas_filepath = os.path.join(self.section_savepath,"atlas_labeled.png")
        if st_switches.section_QL_on:
            labeled_atlas_filepath = os.path.join(self.section_savepath,"tilted_atlas.png")
        elif st_switches.segmentation_1_20_on:
            labeled_atlas_filepath = os.path.join(self.section_savepath,"segmented_atlas.png")
        unlabeled_atlas_filepath = os.path.join(self.section_savepath,"atlas_unlabeled.png")

        mappedatlas_detection = cv.imread(unlabeled_atlas_filepath)
        mappedatlas_unlabled_showimg = cv.imread(unlabeled_atlas_filepath)
        mappedatlas_labled_showimg = cv.imread(labeled_atlas_filepath)

        num_red_blobs = len(red_blobs_modified)
        num_gr_blobs = len(green_blobs_modified)
        atlas_width = mappedatlas_labled_showimg.shape[1]

        redpointcolors = []
        redpointtags = []

        atlas_pil = Image.open(unlabeled_atlas_filepath)
        atlas_colors = Counter(atlas_pil.getdata())
        red_blobs_coords = red_blobs_modified #(c,r

        for point in red_blobs_coords:
            co2, ro2 = point  # Level 3  c, r = xo1, yo1
            bb,gg,rr = mappedatlas_detection[ro2, co2]
            pointcolor = (bb,gg,rr) 
            pointcolor_rgb = (rr,gg,bb) 
            cv.circle(mappedatlas_unlabled_showimg, (co2, ro2), 4, (0, 0, 255), -1)
            cv.circle(mappedatlas_unlabled_showimg, (co2, ro2), 4, (0, 0, 0), 1)
            cv.circle(mappedatlas_labled_showimg, (co2, ro2), 4, (0, 0, 255), -1)
            cv.circle(mappedatlas_labled_showimg, (co2, ro2), 4, (0, 0, 0), 1)
            redpointcolors.append(pointcolor_rgb)
            colorindex = coords_to_colorindex(pointcolor_rgb , self)
            if colorindex==0:
                region = mappedatlas_detection[ro2-regmargin:ro2+regmargin, co2-regmargin:co2+regmargin]
                pointcolor2 = get_region_color(regmargin, region)
                colorindex = coords_to_colorindex(pointcolor2 , self )
            if co2 <= int(atlas_width/2):
                redpointtags.append((colorindex,1))  ## 1 for left side
            if co2 > int(atlas_width/2):
                redpointtags.append((colorindex,2))  ## 2 for right side
        segcountedr = Counter(redpointtags)
        cv.imwrite(os.path.join(self.section_savepath, "mappedatlas_unlabled_showimg.jpg"), mappedatlas_unlabled_showimg)
        blobs_coords_registered = {'red': red_blobs_modified, 'green': green_blobs_modified, 'coloc': colocalized_blobs_coords}
        
        self.saved_data_pickle['blobs_coords_registered'] = blobs_coords_registered
        reportfile = open(os.path.join(self.section_savepath, "reportfile.txt"), 'w')
        reportfile.write('{} Red Blobs in:\n'.format(len(red_blobs_modified)))
        reportfile.write('\n')

        dict_red  = {'type': 'Red', 'Total': num_red_blobs}
        for colortag, count in segcountedr.items():
            pointcolor = self.Bgr_Color_list[colortag[0]]
            if colortag[1]==1:
                label = Regions_n_colors_list[colortag[0]][-3] + " _L"
            elif colortag[1]==2:
                label = Regions_n_colors_list[colortag[0]][-3] + " _R"       

            dict_red[label] = count
            reportfile.write(label + '\t' + str(count) + '\n')
            
        reportfile.write('\n')
        row_red = dict(list(dict_base.items()) + list(dict_red.items())+list({'id':1}.items()))
        
        for regname in regs_per_section[int(atlasnum)]:
            regname_l = regname + " _L"
            regname_r = regname + " _R"
            if regname_l not in row_red:
                row_red[regname_l]=0
            if regname_r not in row_red:
                row_red[regname_r]=0

        self.Report_subdf = self.Report_subdf.append(row_red, ignore_index=True)
        green_blobs_coords = green_blobs_modified #(c,r)

        greenpointcolors = []
        greenpointtags = []
        for point in green_blobs_coords:
            co2, ro2 = point  # bLevel
            bb,gg,rr = mappedatlas_detection[ro2, co2]
            pointcolor = (bb,gg,rr)
            pointcolor_rgb = (rr,gg,bb) 
            cv.circle(mappedatlas_unlabled_showimg, (co2, ro2), 4, (0, 255, 0), -1)
            cv.circle(mappedatlas_unlabled_showimg, (co2, ro2), 4, (0, 0, 0), 1)
            cv.circle(mappedatlas_labled_showimg, (co2, ro2), 4, (0, 255, 0), -1)
            cv.circle(mappedatlas_labled_showimg, (co2, ro2), 4, (0, 0, 0), 1)
            greenpointcolors.append(pointcolor_rgb)
            colorindex = coords_to_colorindex(pointcolor_rgb , self)
            if colorindex==0:
                region = mappedatlas_detection[ro2-regmargin:ro2+regmargin, co2-regmargin:co2+regmargin]
                pointcolor2 = get_region_color(regmargin, region) #RGB
                colorindex = coords_to_colorindex(pointcolor2 , self)
                #print("+++",pointcolor2, colorindex)
            if co2 <= int(atlas_width/2):
                greenpointtags.append((colorindex,1))  ## 1 for left side
            if co2 > int(atlas_width/2):
                greenpointtags.append((colorindex,2))  ## 2 for right side
        segcountedg = Counter(greenpointtags)
        reportfile.write('{} Green Blobs in:\n'.format(len(green_blobs_modified)))
        reportfile.write('\n')

        dict_green = {'type': 'Green', 'Total': num_gr_blobs}
        dict_regs_n_colors_g = {}
        for colortag, count in segcountedg.items():
            pointcolor = self.Bgr_Color_list[colortag[0]]
            if colortag[1]==1:
                label = Regions_n_colors_list[colortag[0]][-3] + " _L"
            elif colortag[1]==2:
                label = Regions_n_colors_list[colortag[0]][-3] + " _R"               

            dict_regs_n_colors_g[label] = [pointcolor, count]
            reportfile.write(label + '\t' + str(count) + '\n')
            dict_green[label] = count
        row_green = dict(list(dict_base.items()) + list(dict_green.items())+list({'id':2}.items()))
        for regname in regs_per_section[int(atlasnum)]:
            regname_l = regname + " _L"
            regname_r = regname + " _R"
            if regname_l not in row_green:
                row_green[regname_l]=0
            if regname_r not in row_green:
                row_green[regname_r]=0

        self.Report_subdf = self.Report_subdf.append(row_green, ignore_index=True)

        colocalized_blobs = colocalized_blobs_coords
        matchcount = len(colocalized_blobs)
        blob_colocs = np.array(colocalized_blobs)
        reportfile.write('\n')
        reportfile.write('{} Co-localization in:\n'.format(matchcount))
        matchpointtags = []
        if matchcount > 0:
            for point in colocalized_blobs:
                co2, ro2 = point
                cv.circle(mappedatlas_unlabled_showimg, (co2, ro2), 5, (0, 255, 255), -1)
                cv.circle(mappedatlas_unlabled_showimg, (co2, ro2), 5, (0, 150, 150), 1)
                cv.circle(mappedatlas_labled_showimg, (co2, ro2), 4, (0, 255, 255), -1)
                cv.circle(mappedatlas_labled_showimg, (co2, ro2), 4, (0, 0, 0), 1)
                bb,gg,rr = mappedatlas_detection[ro2, co2]
                pointcolor = (bb,gg,rr)
                pointcolor_rgb = (rr,gg,bb) 
                colorindex = coords_to_colorindex(pointcolor_rgb , self)
                #colorindex = recheck_colorindex(colorindex, mappedatlas_detection, yo2, xo2)
                if colorindex == 0:
                    region = mappedatlas_detection[ro2-regmargin:ro2+regmargin, co2-regmargin:co2+regmargin]
                    pointcolor2 = get_region_color(regmargin, region)
                    colorindex = coords_to_colorindex(pointcolor2 , self)
                if co2 <= int(atlas_width/2):
                    matchpointtags.append((colorindex,1))  ## 1 for left side
                if co2 > int(atlas_width/2):
                    matchpointtags.append((colorindex,2))  ## 2 for right side
            segcountedm = Counter(matchpointtags)
        else:
            segcountedm = {}

        reportfile.write('\n')
        dict_co = {'type': 'CoLoc', 'Total': matchcount}

        if len(matchpointtags)>0:
            for colortag, count in segcountedm.items():
                pointcolor = self.Bgr_Color_list[colortag[0]]
                if colortag[1]==1:
                    label = Regions_n_colors_list[colortag[0]][-3] + " _L"
                elif colortag[1]==2:
                    label = Regions_n_colors_list[colortag[0]][-3] + " _R"      

                reportfile.write(label + '\t' + str(count) + '\n')
                dict_co[label] = count
        row_coloc = dict(list(dict_base.items()) + list(dict_co.items())+list({'id':3}.items()))
        for regname in regs_per_section[int(atlasnum)]:
            regname_l = regname + " _L"
            regname_r = regname + " _R"
            if regname_l not in row_coloc:
                row_coloc[regname_l]=0
            if regname_r not in row_coloc:
                row_coloc[regname_r]=0
        self.Report_subdf = self.Report_subdf.append(row_coloc, ignore_index=True)

        dict_density = {'type': 'Density', 'Total': '__'}
        dict_area = {'type': 'Area', 'Total': '__'}

        for regname, value  in dict_regs_n_colors_g.items():
            if 'not detected' not in regname:
                region_cfos_count = value[1] 
                region_area = atlas_colors[value[0]]
                region_density =  region_cfos_count / region_area 
                dict_area[regname] = region_area 
                dict_density[regname] = region_density

        row_area = dict(list(dict_base.items()) + list(dict_area.items())+list({'id':4}.items()))
        self.Report_subdf = self.Report_subdf.append(row_area, ignore_index=True)
        row_density = dict(list(dict_base.items()) + list(dict_density.items())+list({'id':5}.items()))
        self.Report_subdf = self.Report_subdf.append(row_density, ignore_index=True)
        if 'id' in self.Report_df:
            self.Report_df = self.Report_df.set_index('id')

        self.Report_subdf = self.Report_subdf.set_index('id')
        for i in range (1,6):
            self.Report_df.loc[((self.brnum-1)*num_rows+i)+8] = self.Report_subdf.loc[i]

        try:
            writer = pd.ExcelWriter(os.path.join(self.savepath, f'Report_{self.slidename}.xlsx'), engine='openpyxl')
            self.Report_df.to_excel(writer, sheet_name='Sheet 1', index=False)
            writer.save()
            writer = pd.ExcelWriter(os.path.join(self.section_savepath, f'Report_{self.slidename}_S{self.brnum}.xlsx'), engine='openpyxl')
            self.Report_subdf.to_excel(writer, sheet_name='Sheet 1', index=False)
            writer.save()
        except:
            return self.section_savepath, "em2"

        #np.save(os.path.join(self.section_savepath, 'bloblocs_co_modified.npy'), blob_colocs)
        reportfile.write(f'\n \n Atlas number: {atlasnum} ')
        reportfile.close()
        cv.imwrite(os.path.join(self.section_savepath, "Analysis_labeled.jpg"), mappedatlas_labled_showimg)
        cv.imwrite(os.path.join(self.section_savepath, "Analysis_unlabeled.jpg"), mappedatlas_unlabled_showimg)

        analyzedimgpath = os.path.join(self.section_savepath, "Analysis_labeled.jpg")
        return self.section_savepath, analyzedimgpath
    
    def get_levels_n_factors(self): 
        return MARGIN, dfba, self.section_savepath
