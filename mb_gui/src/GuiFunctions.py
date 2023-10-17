import numpy as np
import slideio
import pandas as pd
import os,sys
from AMBIA_M1_CellDetection import pool_cell_detection,double_pool_cell_detection,MoG_detection,calculate_colocalized_blobs
from multiprocessing import Pool
import cv2 as cv
import multiprocessing
from skimage.feature import blob_log
from collections import Counter
from math import sqrt
from easysettings import EasySettings
from utils.img_processing import rotate_img, equalize_img
from utils.reading_czi import CZI, czi_channel_regulator,histogram_equalization#, get_channel_info,czi_preview,czi_section_img
import Switches_Static as st_switches
import Switches_Dynamic as dy_switches
from PIL import Image
from collections import Counter
import pickle
import shutil
import time
from utils.allen_functions import high_to_low_level_regions
import Path_Finder
import concurrent.futures







settings = EasySettings("myconfigfile.conf")
if st_switches.atlas_type == "Adult":
    from regionscode.Regions_n_colors_adult import Region_names, general_Region_names, create_regs_n_colors_per_sec_list
    from regionscode.regions_per_sections_adult import regs_per_section


elif st_switches.atlas_type == "Rat":
    from regionscode.Regions_n_colors_rat import Region_names, general_Region_names, create_regs_n_colors_per_sec_list
    from regionscode.regions_per_sections_rat import regs_per_section
else :
    print ("Incorrect atlas type, exiting.")
    sys.exit()


MARGIN = st_switches.MARGIN
num_rows = st_switches.num_rows

saved_data_pickle = {}


def write_registration_image(path, img):
    range = np.amax(img) - np.amin(img)
    img = img - np.amin(img)
    img2 = (img / range) * 255
    cv.imwrite(path, img2)
    return


def get_region_color(regmargin, section):
    sectioncolors = []
    for i in range (0,regmargin*2):
        for j in range(0,regmargin*2):
            b,g,r = section[i,j]
            sectioncolors.append((r,g,b))
    pointcolor2 = max(set(sectioncolors), key = sectioncolors.count)
    return pointcolor2
rootpath=Path_Finder.return_root_path()

#OPENSLIDE_PATH = "F:\AMBIA\git_lab_rat_version_branch\mb_gui\openslide_dlls"
os.path.join(rootpath, "Gui_Atlases", "Adult_full_atlases")
OPENSLIDE_PATH = os.path.join(rootpath, "mb_gui","openslide_dlls")
print (OPENSLIDE_PATH)
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

def save_to_saved_data_pickle(item, keytext):

    saved_data_pickle[keytext] = item


def funcLandmarkDetection(imgpath, midheight):
    # midh = 200 for tissue
    # midh = 400 for atlas
    LandmarksT = []
    return LandmarksT

class Slide_Operator:
    def __init__(self,prepath):
        self.prepath=prepath
    def funcReadSlide(self,slide_fullpath):
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
        self.report_xls_path = os.path.join(self.savepath, f'Report_{self.slidename}.xlsx')
        
        print('Slide path: ', slide_fullpath)
        if self.slideformat == "mrxs":
            print("It is mirax format")
            self.mlevel = st_switches.mrx_mlevel  # mask level
            self.blevel = st_switches.mrx_blevel  # Blob detection level
            self.alevel = st_switches.mrx_alevel  # Atlas mapping level
            Slide = openslide.OpenSlide(slide_fullpath)
            Dims = Slide.level_dimensions
            slideimage = Slide.read_region((0, 0), self.mlevel, (Dims[self.mlevel][0], Dims[self.mlevel][1]))
            slideimage = cv.cvtColor(np.array(slideimage), cv.COLOR_RGB2BGR)
            slideimage_equalized = equalize_img(slideimage)
            slideimage = cv.rotate(slideimage_equalized, cv.ROTATE_90_CLOCKWISE)
            slideimgpath = os.path.join(self.savepath, "mlevel.jpg")
            cv.imwrite(slideimgpath, slideimage)
            self.num_channels = st_switches.mrx_num_channels
            self.channel_types = ["B", "G", "R"]
            self.maskthresh = st_switches.mrx_maskthresh
        elif self.slideformat == "czi":
            self.czi=CZI(slide_fullpath)
            print("It is czi format")
            self.mlevel = st_switches.czi_mlevel  # mask level
            self.blevel = st_switches.czi_blevel  # Blob detection level
            self.alevel = st_switches.czi_alevel  # Atlas mapping level
            slideimgpath = os.path.join(self.savepath, "mlevel.jpg")
            self.num_channels, self.channel_types = self.czi.get_channel_info(slide_fullpath)
            self.czi.czi_preview(slide_fullpath, slideimgpath, self.mlevel)
            self.maskthresh = st_switches.czi_maskthresh
        else:
            print("Unsupported slide format")

        self.dfm0 = 2 ** self.mlevel  # Convert factor from mask level to level zero
        self.dfma = 2 ** (self.mlevel - self.alevel)
        self.dfmb = 2 ** (self.mlevel - self.blevel)
        self.dfba = 2 ** (self.alevel - self.blevel)
        return slideimgpath
    
    def funcSectionDetection(self):
        time1=time.time()
        #global Report_df
        brainboundcoords = np.array([[0, 0, 0, 0]])  #[x, y, w, h]
        slideimg = cv.imread(os.path.join(self.savepath, "mlevel.jpg"))

        print (os.path.join(self.savepath, "mlevel.jpg"))
        imgray = cv.cvtColor(slideimg, cv.COLOR_BGR2GRAY)
        img_gr_mblur = cv.medianBlur(imgray, 9)
        # In mask level specified above
        _, slidemask = cv.threshold(img_gr_mblur, self.maskthresh, 255, cv.THRESH_BINARY)  # mlevel
        cv.imwrite(os.path.join(self.savepath, "tissue_mask.jpg"), slidemask)
        cv.imwrite(os.path.join(self.savepath, "imgray.jpg"), imgray)
        contours, _ = cv.findContours(slidemask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # mlevel
        
        threshold_area = 10000
        areas = np.array([cv.contourArea(c) for c in contours])
        
        small_contours_indexes = np.argwhere(areas < threshold_area)
        contours = [contours[i] for i in range(len(contours)) if i not in small_contours_indexes]
        
        contours=list(contours)

        
        br_min_coords = [(x, y) for x, y, w, h in (cv.boundingRect(c) for c in contours)]
        br_min_coords_by5 = [x + y * 5 for x, y in br_min_coords]
        br_min_coords_by5_sorted = sorted(br_min_coords_by5)
        rearrange_guide = [(i, j) for i, coord in enumerate(br_min_coords_by5) for j, coord_sorted in enumerate(br_min_coords_by5_sorted) if coord == coord_sorted]
        rearrange_guide_sorted = sorted(rearrange_guide, key=lambda tup: tup[1])

        brnumtemp = 1
        for indextuple in rearrange_guide_sorted:
            indexnew = indextuple[0]
            cnt = contours[indexnew]

            x, y, w, h = cv.boundingRect(cnt)  # mlevel        
            brainboundcoords = np.append(brainboundcoords, [[x, y, w, h]], axis=0)
            brnumtemp += 1  


        
        if self.slideformat == "mrxs":
            tissuemask_fullpath = os.path.join(self.savepath, "imgray.jpg")
        elif self.slideformat == "czi":
            tissuemask_fullpath = os.path.join(self.savepath, "imgray.jpg")

                
        brainboundcoords = brainboundcoords[1:, :]
        time2=time.time()
        print(f"SectionDetection took {time2-time1}")
        return brainboundcoords, tissuemask_fullpath
    


    def get_section_images(self,brnum0, brainboundcoords):
        num_processes = 5  # Adjust this to the number of processes you want
        pool = multiprocessing.Pool(processes=num_processes)
        time3=time.time()

        self.section_savepath = os.path.join(self.savepath, f"S{brnum0}")
        
        if not os.path.exists(self.section_savepath):
            os.makedirs(self.section_savepath)
        if self.slideformat == "mrxs":
            x, y, w, h = brainboundcoords[brnum0-1]  # mlevel
            [xb, yb, wb, hb] = [x * self.dfmb, y * self.dfmb, w * self.dfmb, h * self.dfmb]
            [xa, ya, wa, ha] = [x * self.dfma, y * self.dfma, w * self.dfma, h * self.dfma]
            Slide = openslide.OpenSlide(self.slidepath)
            Dims = Slide.level_dimensions
            brainimg = Slide.read_region((y * self.dfm0, Dims[0][1] - ((x + w) * self.dfm0)), self.blevel, (hb, wb)).convert("RGB")
            brainimg2 = np.array(brainimg)
            section_blevel = cv.rotate(brainimg2, cv.ROTATE_90_CLOCKWISE)
            
            if st_switches.color_switch_on:
                section_blevel2 = section_blevel.copy()
                section_blevel2[:,:,1]= section_blevel[:,:,2]
                section_blevel2[:,:,2]= section_blevel[:,:,1]
                section_blevel = section_blevel2
            section_blevel_eq = equalize_img(section_blevel)
            braina = Slide.read_region((y * self.dfm0, Dims[0][1] - ((x + w) * self.dfm0)), self.alevel, (ha, wa)).convert("RGB")
            braina_dark = np.array(braina)
            braina_rot = cv.rotate(braina_dark, cv.ROTATE_90_CLOCKWISE)
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
            section_alevel = self.czi.czi_section_img(self.slidepath, brnum0, num_sections, self.alevel, [], rect=None)
            section_blevel = self.czi.czi_section_img(self.slidepath, brnum0, num_sections, self.blevel, [], rect=None)
            
            section_alevel = czi_channel_regulator(section_alevel)
            section_blevel = czi_channel_regulator(section_blevel)
            section_alevel = cv.copyMakeBorder(section_alevel, MARGIN, MARGIN, MARGIN, MARGIN, cv.BORDER_CONSTANT,
                                            value=(0, 0, 0))
            section_alevel_eq = histogram_equalization(section_alevel)
            section_blevel_eq = histogram_equalization(section_blevel) 
            #pool.apply_async(cv.imwrite, (os.path.join(self.section_savepath,"alevel.png"), cv.rotate(section_alevel, cv.ROTATE_90_CLOCKWISE)))

            if st_switches.rotate_flag:
                pool.apply_async(cv.imwrite, (os.path.join(self.section_savepath,"alevel.png"), cv.rotate(section_alevel, cv.ROTATE_90_CLOCKWISE)))
                pool.apply_async(cv.imwrite, (os.path.join(self.section_savepath,"blevel.png"), cv.rotate(section_blevel, cv.ROTATE_90_CLOCKWISE)))
                pool.apply_async(cv.imwrite, (os.path.join(self.section_savepath,"alevel_eq.png"), cv.rotate(section_alevel_eq, cv.ROTATE_90_CLOCKWISE)))
                pool.apply_async(cv.imwrite, (os.path.join(self.section_savepath,"blevel_eq.png"), cv.rotate(section_blevel_eq, cv.ROTATE_90_CLOCKWISE)))
                """cv.imwrite(os.path.join(self.section_savepath,"alevel.png"), cv.rotate(section_alevel, cv.ROTATE_90_CLOCKWISE))
                cv.imwrite(os.path.join(self.section_savepath,"blevel.png"), cv.rotate(section_blevel, cv.ROTATE_90_CLOCKWISE))
                cv.imwrite(os.path.join(self.section_savepath,"alevel_eq.png"), cv.rotate(section_alevel_eq, cv.ROTATE_90_CLOCKWISE))
                cv.imwrite(os.path.join(self.section_savepath,"blevel_eq.png"), cv.rotate(section_blevel_eq, cv.ROTATE_90_CLOCKWISE))"""
            else:
                """cv.imwrite(os.path.join(self.section_savepath,"alevel.png"), section_alevel)
                cv.imwrite(os.path.join(self.section_savepath,"blevel.png"), section_blevel)
                cv.imwrite(os.path.join(self.section_savepath,"alevel_eq.png"), section_alevel_eq)
                cv.imwrite(os.path.join(self.section_savepath,"blevel_eq.png"), section_blevel_eq)"""
                pool.apply_async(cv.imwrite,(os.path.join(self.section_savepath,"alevel.png"), section_alevel))
                pool.apply_async(cv.imwrite,(os.path.join(self.section_savepath,"blevel.png"), section_blevel))
                pool.apply_async(cv.imwrite,(os.path.join(self.section_savepath,"alevel_eq.png"), section_alevel_eq))
                pool.apply_async(cv.imwrite,(os.path.join(self.section_savepath,"blevel_eq.png"), section_blevel_eq))
            for channel in range(self.num_channels):
                #channel_name = self.channel_types[channel]
                blevel_channel = self.czi.czi_section_img(self.slidepath, brnum0, num_sections, self.blevel, [channel], rect=None)
 
                if st_switches.rotate_flag:
                    cv.imwrite(os.path.join(self.section_savepath, f"blevel_{channel}.png"), cv.rotate(blevel_channel, cv.ROTATE_90_CLOCKWISE))
                else : 
                    cv.imwrite(os.path.join(self.section_savepath, f"blevel_{channel}.png"), blevel_channel)
            


        
        blob_detection_file_name = os.path.join(self.section_savepath,"blevel_eq.png")
        tissue_lm_detection_filename = os.path.join(self.section_savepath,"alevel_eq.png")
        
        time4=time.time()

        
        pool.close()
        pool.join()
        print(f"Getting Section Images took {time4-time3}")
        return blob_detection_file_name, tissue_lm_detection_filename
    
    
    def funcBlobDetection(self, brnum, blobs_parameters):
        """ Returns blobs_log_r, blobs_log_g, colocalized_blobs:: list of blob coords (r,c) before 
        adding the blobs added/removed manually by user
        Saves blobs_locs_r, blobs_locs_g, blob_locs_co as numpy.array
        this also include blob coords (r,c) before adding the blobs added/removed manually by user
        Saves  r_params_for_save, g_params_for_save as npy
        """

        tempMARGIN = 50  # temporary margin just to avoid the borders when applying thresh, adding the margin is reversed in the parameter brain_mask_eroded

        brain_blevel = cv.imread(os.path.join(self.section_savepath, 'blevel_eq.png'))
        brainimgtemp = cv.copyMakeBorder(brain_blevel, tempMARGIN, tempMARGIN, tempMARGIN, tempMARGIN, cv.BORDER_CONSTANT, value=(0, 0, 0))
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
        #cv.imwrite(os.path.join(section_savepath, 'brain_mask_closed.jpg'), closing)
        brain_mask_eroded_uncut = cv.erode(closing, kernel2, iterations=3)
        #cv.imwrite(os.path.join(section_savepath, 'brain_mask_eroded.jpg'), brain_mask_eroded_uncut)

        brain_mask_eroded = brain_mask_eroded_uncut[tempMARGIN:-tempMARGIN, tempMARGIN:-tempMARGIN]
        cv.imwrite(os.path.join(self.section_savepath, 'brain_mask_eroded_cut.jpg'), brain_mask_eroded)

        #img_channel_g = cv.medianBlur(img_channel_g, 3)
        #img_channel_g = equalize_img(img_channel_g)
        # hist,bins = np.histogram(img_channel_g.flatten(),256,[0,256])
        # cdf = hist.cumsum()
        # cdf_normalized = cdf * hist.max()/ cdf.max()
        # cdf_m = np.ma.masked_equal(cdf,0)
        # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        # cdf = np.ma.filled(cdf_m,0).astype('uint8')
        # img_channel_g = cdf[img_channel_g]
        
        #img_not_r = cv.medianBlur(img_channel_r, 3)

        ### Parameters
        red_blob_type = blobs_parameters["red_blob_type"]
        green_blob_type = blobs_parameters["green_blob_type"]
        green_blobs_thresh = blobs_parameters["green_blob_thresh"]
        number_of_blobs_g = 0
        number_of_blobs_r = 0
        blobs_parameters_dict_to_save = {}
        ######### Red blobs detection
    
        if red_blob_type == "cFos" and green_blob_type == "cFos":
            minsigma = 1
            thresh_r = blobs_parameters['red_blob_min_sigma']
            maxsigma_r = blobs_parameters['red_blob_max_sigma']
            numsigma_r = blobs_parameters['red_blob_num_sigma']
            red_blobs_thresh = blobs_parameters['red_blob_thresh2'] /100
            _, ch_thresh = cv.threshold(img_channel_r, thresh_r, 255, cv.THRESH_BINARY)
            img_channel_r = cv.bitwise_and(img_channel_r, img_channel_r, mask = ch_thresh)
            
            blobs_parameters_dict_to_save['red'] = [thresh_r, maxsigma_r, numsigma_r, red_blobs_thresh]
            thresh_g = blobs_parameters['green_blob_min_sigma']
            maxsigma_g = blobs_parameters['green_blob_max_sigma']
            numsigma_g = blobs_parameters['green_blob_num_sigma']
            green_blobs_thresh = blobs_parameters['green_blob_thresh2']/100
            
            _, ch_thresh = cv.threshold(img_channel_g, thresh_g, 255, cv.THRESH_BINARY)
            
            img_channel_g = cv.bitwise_and(img_channel_g, img_channel_g, mask = ch_thresh)
            
            #blobs_log_g = pool_cell_detection(img_channel_g, brain_mask_eroded, thresh_g, maxsigma_g, numsigma_g, green_blobs_thresh, "green_cells")
            blobs_parameters_dict_to_save['green'] = [thresh_g, maxsigma_g, numsigma_g, green_blobs_thresh]

            

            self.blobs_log_r, self.blobs_log_g, matchcount, blob_locs_co = double_pool_cell_detection(img_channel_r, img_channel_g, brain_mask_eroded, minsigma, maxsigma_r, numsigma_r, red_blobs_thresh, maxsigma_g, numsigma_g, green_blobs_thresh)
            
        else:
            if red_blob_type == "Rabies":
                minsize = blobs_parameters['red_blob_min_size']
                red_blobs_thresh = blobs_parameters["red_blob_thresh"]
                self.blobs_log_r = self.rabies_detection(img_channel_r, red_blobs_thresh, minsize, brain_mask_eroded)
                #r_params_for_save = np.array([minsize,red_blobs_thresh])
                blobs_parameters_dict_to_save['red'] = [minsize, red_blobs_thresh]
                #np.save(os.path.join(section_savepath, 'blobparams_r.npy'), r_params_for_save)

            if red_blob_type == "MoG":
                min_corr = blobs_parameters['red_blob_correlation']
                stride = blobs_parameters['red_blob_stride']
                self.blobs_log_r = MoG_detection(img_channel_r, min_corr, stride, brain_mask_eroded)

            elif red_blob_type == "cFos":
                minsigma = 1
                r_thresh = blobs_parameters['red_blob_min_sigma']
                maxsigma = blobs_parameters['red_blob_max_sigma']
                numsigma = blobs_parameters['red_blob_num_sigma']
                red_blobs_thresh = blobs_parameters['red_blob_thresh2'] /100

                ret, ch_thresh = cv.threshold(img_channel_r, r_thresh, 255, cv.THRESH_BINARY)
                img_channel_r = cv.bitwise_and(img_channel_r, img_channel_r, mask = ch_thresh)

                #blobs_log_r = cfos_detection(img_channel_r, minsigma, maxsigma, numsigma, red_blobs_thresh, brain_mask_eroded)
                self.blobs_log_r = pool_cell_detection(img_channel_r, brain_mask_eroded, minsigma, maxsigma, numsigma, red_blobs_thresh, "red_cells")
                blobs_parameters_dict_to_save['red'] = [minsigma, maxsigma, numsigma, red_blobs_thresh]

            ####### Green blobs detection
            if green_blob_type == "Rabies":
                minsize = blobs_parameters['green_blob_min_size']
                green_blobs_thresh = blobs_parameters['green_blob_thresh']
                self.blobs_log_g = self.rabies_detection(img_channel_g, green_blobs_thresh, minsize, brain_mask_eroded)
                #g_params_for_save = np.array()
                blobs_parameters_dict_to_save['green'] = [minsize,green_blobs_thresh]

            elif green_blob_type == "MoG":
                min_corr = blobs_parameters['green_blob_correlation']
                stride = blobs_parameters['green_blob_stride']
                self.blobs_log_g = MoG_detection(img_channel_g, min_corr, stride, brain_mask_eroded)

            elif green_blob_type == "cFos":
                minsigma = 1
                g_thresh = blobs_parameters['green_blob_min_sigma']
                maxsigma = blobs_parameters['green_blob_max_sigma']
                numsigma = blobs_parameters['green_blob_num_sigma']
                green_blobs_thresh = blobs_parameters['green_blob_thresh2']/100

                _, ch_thresh = cv.threshold(img_channel_g, g_thresh, 255, cv.THRESH_BINARY)
                img_channel_g = cv.bitwise_and(img_channel_g, img_channel_g, mask = ch_thresh)

                self.blobs_log_g = pool_cell_detection(img_channel_g, brain_mask_eroded, minsigma, maxsigma, numsigma, green_blobs_thresh, "green_cells")


                blobs_parameters_dict_to_save['green'] = [minsigma, maxsigma, numsigma, green_blobs_thresh]
                #np.save(os.path.join(section_savepath, 'blobparams_g.npy'), g_params_for_save)
            matchcount, blob_locs_co = calculate_colocalized_blobs(self.blobs_log_r, self.blobs_log_g)

        
        number_of_blobs_g = len(self.blobs_log_g)
        number_of_blobs_r = len(self.blobs_log_r)
        #save_to_pkl("blobs_parameters", blobs_parameters)
        saved_data_pickle['blobs_parameters'] = blobs_parameters_dict_to_save
        ####### colocalized
        """
        brain_blevel1 = brain_blevel.copy()

        for point in blobs_log_r:
            co2, ro2 = point  # Level 3  c, r = xo1, yo1
            cv.circle(brain_blevel1, (ro2, co2), 7, (0, 0, 255), 1)

        brain_blevel2 = brain_blevel.copy()

        for point in blobs_log_g:
            co2, ro2 = point  # Level 3  c, r = xo1, yo1
            cv.circle(brain_blevel2, (ro2, co2), 7, (0, 255, 0), 1)
        
        cv.imwrite(os.path.join(section_savepath, "cfos_detected_red.png"), brain_blevel1)
        cv.imwrite(os.path.join(section_savepath, "cfos_detected_green.png"), brain_blevel2)"""
        screenimg_path = os.path.join(self.section_savepath, 'blevel_eq.png')
        #np.save(os.path.join(section_savepath, "bloblocs_g_auto.npy"), blob_locs_g)
        #np.save(os.path.join(section_savepath, "bloblocs_r_auto.npy"), blob_locs_r)
        #np.save(os.path.join(section_savepath, "bloblocs_co_auto.npy"), blob_locs_co)
        return number_of_blobs_r, number_of_blobs_g, matchcount, screenimg_path, self.blobs_log_r, self.blobs_log_g, blob_locs_co
    def funcAnalysis(self,atlasnum, brnum, atlas_prepath, red_blobs_modified, green_blobs_modified, colocalized_blobs_coords):
        """ Inputs red/green_blobs_modified as a list of blob coords (c, r)
        these include detected blob coords after user modification
        red/green_blobs_modified coords are in blevel
        Intermediate variable blob_locs_r/g np array  [r,c]
        """
        #global Report_df
        #global Report_subdf
        #global Bgr_Color_list
        #global Rgb_Color_list
        #global saved_data_pickle
        regmargin = 5  #for color averaging
        if os.path.exists(self.report_xls_path):
            self.Report_df = pd.read_excel(self.report_xls_path)
            print("xlsx exists")
            self.report_flag=1
        else:
            self.report_flag=0
        if self.slideformat == "mrxs":
            dict_base = {'Animal': self.Experiment_num, 'Rack': self.rack_num, 'Slide': self.slide_num, 'Section': brnum}
            Report_subdf = pd.DataFrame(columns=['id', 'Animal', 'Rack', 'Slide', 'Section', 'type', 'Total'] + Region_names)
        elif self.slideformat == "czi":  
            dict_base = {'Experiment': self.Experiment_num, 'Animal': self.rack_num, 'Slide': self.slide_num, 'Section': brnum}
            
            Report_subdf = pd.DataFrame(columns=['id', 'Experiment', 'Animal', 'Slide', 'Section', 'type', 'Total'] + Region_names)
        Regions_n_colors_list, Bgr_Color_list, self.Rgb_Color_list =  create_regs_n_colors_per_sec_list(atlasnum)

        savepath = os.path.join(self.prepath, self.slidename)
        labeled_atlas_filepath = os.path.join(atlas_prepath,"labeled_atlases", str(atlasnum)+".png")
        if st_switches.section_QL_on:
            labeled_atlas_filepath = os.path.join(self.section_savepath,"tilted_atlas.png")
        elif st_switches.segmentation_1_20_on:
            labeled_atlas_filepath = os.path.join(self.section_savepath,"segmented_atlas.png")
        unlabeled_atlas_filepath = os.path.join(atlas_prepath,"unlabeled_atlases", str(atlasnum)+".png")

        mappedatlas_detection = cv.imread(unlabeled_atlas_filepath)
        mappedatlas_unlabled_showimg = cv.imread(unlabeled_atlas_filepath)
        mappedatlas_labled_showimg = cv.imread(labeled_atlas_filepath)

        num_red_blobs = len(red_blobs_modified)
        num_gr_blobs = len(green_blobs_modified)
        atlas_width = mappedatlas_labled_showimg.shape[1]

        redpointcolors = []
        redpointtags = []

        atlas_pil = Image.open(unlabeled_atlas_filepath).convert('RGB')
        atlas_colors = Counter(atlas_pil.getdata())
        red_blobs_coords = red_blobs_modified #(c,r
        #print (red_blobs_coords)
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
            colorindex = self.coords_to_colorindex(pointcolor_rgb)
            if colorindex==0:
                region = mappedatlas_detection[ro2-regmargin:ro2+regmargin, co2-regmargin:co2+regmargin]
                pointcolor2 = get_region_color(regmargin, region)
                colorindex = self.coords_to_colorindex(pointcolor2)
            if co2 <= int(atlas_width/2):
                redpointtags.append((colorindex,1))  ## 1 for left side
            if co2 > int(atlas_width/2):
                redpointtags.append((colorindex,2))  ## 2 for right side
        segcountedr = Counter(redpointtags)
        cv.imwrite(os.path.join(self.section_savepath, "mappedatlas_unlabled_showimg.jpg"), mappedatlas_unlabled_showimg)
        blobs_coords_registered = {'red': red_blobs_modified, 'green': green_blobs_modified, 'coloc': colocalized_blobs_coords}
        
        saved_data_pickle['blobs_coords_registered'] = blobs_coords_registered
        #save_to_pkl("blobs_coords_registered.pkl", blobs_coords_registered)

        #np.save(os.path.join(section_savepath, "red_blobs_registered.npy"), red_blobs_modified)
        #np.save(os.path.join(section_savepath, "green_blobs_registered.npy"), green_blobs_modified)
        reportfile = open(os.path.join(self.section_savepath, "reportfile.txt"), 'w')
        reportfile.write('{} Red Blobs in:\n'.format(len(red_blobs_modified)))
        reportfile.write('\n')

        dict_red  = {'type': 'Red', 'Total': num_red_blobs}
        for colortag, count in segcountedr.items():
            pointcolor = self.Rgb_Color_list[colortag[0]]
            if colortag[1]==1:
                label = Regions_n_colors_list[colortag[0]][-3] + "_L"
            elif colortag[1]==2:
                label = Regions_n_colors_list[colortag[0]][-3] + "_R"       

            dict_red[label] = count
            reportfile.write(label + '\t' + str(count) + '\n')
            
        reportfile.write('\n')

        row_red = dict(list(dict_base.items()) + list(dict_red.items())+list({'id':1}.items()))
        for regname in regs_per_section[int(atlasnum)]:
            regname_l = regname + "_L"
            regname_r = regname + "_R"
            if regname_l not in row_red:
                row_red[regname_l]=0
            if regname_r not in row_red:
                row_red[regname_r]=0

        temp=Report_subdf

        Report_subdf = Report_subdf._append(row_red,ignore_index=True)
        
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
            colorindex = self.coords_to_colorindex(pointcolor_rgb)
            if colorindex==0:
                region = mappedatlas_detection[ro2-regmargin:ro2+regmargin, co2-regmargin:co2+regmargin]
                pointcolor2 = get_region_color(regmargin, region) #RGB
                colorindex = self.coords_to_colorindex(pointcolor2)
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
            pointcolor = self.Rgb_Color_list[colortag[0]]
            if colortag[1]==1:
                label = Regions_n_colors_list[colortag[0]][-3] + "_L"
            elif colortag[1]==2:
                label = Regions_n_colors_list[colortag[0]][-3] + "_R"               
            dict_regs_n_colors_g[label] = [pointcolor, count]
            reportfile.write(label + '\t' + str(count) + '\n')
            dict_green[label] = count
        row_green = dict(list(dict_base.items()) + list(dict_green.items())+list({'id':2}.items()))
    
        for regname in regs_per_section[int(atlasnum)]:
            regname_l = regname + "_L"
            regname_r = regname + "_R"
            if regname_l not in row_green:
                row_green[regname_l]=0
            if regname_r not in row_green:
                row_green[regname_r]=0

        Report_subdf = Report_subdf._append(row_green, ignore_index=True)

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
                colorindex = self.coords_to_colorindex(pointcolor_rgb)
                #colorindex = recheck_colorindex(colorindex, mappedatlas_detection, yo2, xo2)
                if colorindex == 0:
                    region = mappedatlas_detection[ro2-regmargin:ro2+regmargin, co2-regmargin:co2+regmargin]
                    pointcolor2 = get_region_color(regmargin, region)
                    colorindex = self.coords_to_colorindex(pointcolor2)
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
                pointcolor = self.Rgb_Color_list[colortag[0]]
                if colortag[1]==1:
                    label = Regions_n_colors_list[colortag[0]][-3] + "_L"
                elif colortag[1]==2:
                    label = Regions_n_colors_list[colortag[0]][-3] + "_R"      

                reportfile.write(label + '\t' + str(count) + '\n')
                dict_co[label] = count
        row_coloc = dict(list(dict_base.items()) + list(dict_co.items())+list({'id':3}.items()))
        for regname in regs_per_section[int(atlasnum)]:
            regname_l = regname + "_L"
            regname_r = regname + "_R"
            if regname_l not in row_coloc:
                row_coloc[regname_l]=0
            if regname_r not in row_coloc:
                row_coloc[regname_r]=0


        temp=Report_subdf
        Report_subdf = Report_subdf._append(row_coloc, ignore_index=True)


        dict_density = {'type': 'Density', 'Total': '__'}
        dict_area = {'type': 'Area', 'Total': '__'}

        for regname, value  in dict_regs_n_colors_g.items():
            if 'not detected' not in regname.lower():

                region_cfos_count = value[1] 
                #if st_switches.atlas
                region_area = atlas_colors[value[0]]
                region_density =  region_cfos_count / region_area 
                dict_area[regname] = region_area 
                dict_density[regname] = region_density

        row_area = dict(list(dict_base.items()) + list(dict_area.items())+list({'id':4}.items()))
        Report_subdf = Report_subdf._append(row_area, ignore_index=True)

        row_density = dict(list(dict_base.items()) + list(dict_density.items())+list({'id':5}.items()))
        Report_subdf = Report_subdf._append(row_density, ignore_index=True)

        try: 
            if 'id' in self.Report_df:
                self.Report_df = self.Report_df.set_index('id')
        except : 
            pass
        Report_subdf = Report_subdf.set_index('id')


        if self.report_flag:
            

            if brnum in self.Report_df["Section"].values:
                ms=self.Report_df.index[self.Report_df['Section'] == brnum]
           
                Report_subdf=Report_subdf.set_index(ms)
                self.Report_df.loc[ms]=Report_subdf.loc[:]
        
                #self.Report_df = self.Report_df[self.Report_df['Section'] != brnum]
                #the line above removes the row with same section number
            else :
                print ("section does not exist in report,appending")
                for i in range(Report_subdf.shape[0]):
                    self.Report_df=self.Report_df._append(Report_subdf.iloc[i])
  
        else :
            self.report_flag=True
            self.Report_df=Report_subdf

        try:
            writer = pd.ExcelWriter(os.path.join(savepath, f'Report_{self.slidename}.xlsx'), engine='openpyxl')
            self.Report_df.to_excel(writer, sheet_name='Sheet 1', index=False)
            print ("report created")
            writer.close()
            
            writer = pd.ExcelWriter(os.path.join(self.section_savepath, f'Report_{self.slidename}_S{brnum}.xlsx'), engine='openpyxl')
            Report_subdf.to_excel(writer, sheet_name='Sheet 1', index=False)
            writer.close()
        except Exception as E:
            print( E)
            return self.section_savepath, "em2"
        reportfile.write(f'\n \n Atlas number: {atlasnum} ')
        reportfile.close()
        
        if st_switches.atlas_type == "Rat":
            Report_df_general = self.Report_df
            shutil.copy(os.path.join(self.section_savepath, "reportfile.txt"), os.path.join(self.section_savepath, "reportfile_low.txt"))
        elif st_switches.atlas_type=="Adult":
            Report_df_general = high_to_low_level_regions(self.section_savepath, Region_names, general_Region_names, Report_subdf)
            writer = pd.ExcelWriter(os.path.join(self.section_savepath, f'Report_low__{self.slidename}_S{brnum}.xlsx'), engine='openpyxl')
            Report_df_general.to_excel(writer, sheet_name='Sheet 1', index=False)
            writer.close()

        #np.save(os.path.join(section_savepath, 'bloblocs_co_modified.npy'), blob_colocs)

        cv.imwrite(os.path.join(self.section_savepath, "Analysis_labeled.jpg"), mappedatlas_labled_showimg)
        cv.imwrite(os.path.join(self.section_savepath, "Analysis_unlabeled.jpg"), mappedatlas_unlabled_showimg)

        analyzedimgpath = os.path.join(self.section_savepath, "Analysis_labeled.jpg")
        return self.section_savepath, analyzedimgpath
    

    def rabies_detection(self,img_channel, blobs_thresh, minsize, brain_mask_eroded):
        """
        Uses morphological operations to detect blobs, Outputs blobs_log coords of detected cells in blevel image
        blobs_log: list of coords, (r, c)
        !! blobs_logs only includes list of blobs detected by this function and not the ones added manually by the user
        """
        blobs_log = []
        _, mask = cv.threshold(img_channel, blobs_thresh, 255, cv.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv.erode(mask, kernel, iterations=1)
        cv.imwrite(os.path.join(self.section_savepath, 'red_mask.jpg'), mask_eroded)
        components, _, stats, centers = cv.connectedComponentsWithStats(mask_eroded, 4, cv.CV_32S)

        for compnum in range(1, components):
            size = stats[compnum][-1]
            cb, rb = centers[compnum]
            if brain_mask_eroded[int(rb), int(cb)]==255:
                if size > minsize:
                    blobs_log.append((int(rb), int(cb)))
        return blobs_log

    ###################################################################
    # Detects individual sections
    # Outputs bounding box corrds for each section in brainboundcoords
    # brainboundcoords = list of [x, y, w, h]
    ###################################################################

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

    def save_to_pkl(self,filename, data):
        b_file = open(os.path.join(self.section_savepath, filename), "wb")
        pickle.dump(data, b_file)
        b_file.close()


    def calculate_fp_fn_blobs(self,red_blobs_modified, green_blobs_modified):
        #global saved_data_pickle
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
        saved_data_pickle['blobs_fp_fn'] = blobs_fp_fn
        #save_to_pkl("blobs_coords_fp_fn.pkl", blobs_fp_fn)
        return
    def coords_to_colorindex(self,pointcolor):
        if pointcolor in self.Rgb_Color_list:
            colorindex = self.Rgb_Color_list.index(pointcolor)
        else:
            colorindex = 0
        return colorindex
    def get_levels_n_factors(self):
        return MARGIN, self.dfba, self.section_savepath
    
    def save_the_pickle(self):
        self.save_to_pkl("precious_saved_data.pkl", saved_data_pickle)
        return



