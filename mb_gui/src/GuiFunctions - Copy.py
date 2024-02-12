import numpy as np
from copy import deepcopy
import slideio
import pandas as pd
import os,sys
from AMBIA_M1_CellDetection import pool_cell_detection,MoG_detection,calculate_colocalized_blobs,calc_new_coords,cfos_detection
from multiprocessing import Pool
import cv2 as cv
import multiprocessing
from collections import Counter
from math import sqrt
from easysettings import EasySettings
from utils.img_processing import equalize_img
import utils.img_processing as imgprc
from utils.reading_czi import CZI, czi_channel_regulator,histogram_equalization
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
import time

BlobColor_=[(0,0,255),(0,255,0),(255,0,0),(255,0,255),(0,255,255),(255,255,0),(255,255,255)]

settings = EasySettings("myconfigfile.conf")
if st_switches.atlas_type == "Adult":
    from regionscode.Regions_n_colors_adult import Region_names, general_Region_names, create_regs_n_colors_per_sec_list
    from regionscode.regions_per_sections_adult import regs_per_section


elif st_switches.atlas_type == "Rat":
    from regionscode.Regions_n_colors_rat import Region_names, general_Region_names, create_regs_n_colors_per_sec_list
    from regionscode.regions_per_sections_rat import regs_per_section
else :
    print("Incorrect atlas type, exiting.")
    sys.exit()


MARGIN = st_switches.MARGIN
ALEVEL_MASK_THRESH = st_switches.alevel_mask_threshold
BLEVEL_MASK_THRESH = st_switches.blevel_mask_threshold
CH_O = st_switches.channel_to_omit

saved_data_pickle = {}


def write_registration_image(path, img):
    range = np.amax(img) - np.amin(img)
    img = img - np.amin(img)
    img2 = (img / range) * 255
    cv.imwrite(path, img2)
    return


def get_region_color(regmargin, section):
    try:
        sectioncolors = []
        for i in range (0,regmargin*2):
            for j in range(0,regmargin*2):
                b,g,r = section[i,j]
                sectioncolors.append((r,g,b))
        pointcolor2 = max(set(sectioncolors), key = sectioncolors.count)
    except:
        pointcolor2 = (0,0,0)
    return pointcolor2
rootpath=Path_Finder.return_root_path()

os.path.join(rootpath, "Gui_Atlases", "Adult_full_atlases")
OPENSLIDE_PATH = os.path.join(rootpath, "mb_gui","openslide_dlls")
print(OPENSLIDE_PATH)
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

def save_to_saved_data_pickle(item, keytext):
    saved_data_pickle[keytext] = item




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
            print (slideimage.shape,"slidddddddddddde")
            if st_switches.Bright_field:
                slideimage=(-1)*slideimage
                slideimage=np.apply_along_axis(lambda arr: arr-np.min(arr),arr=slideimage,axis=2)
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
            print ("name of channels",self.channel_types)
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
    
    def remove_edge_blob(self, img, min_contour_area=100, margin=2):
        # Get the contours of all blobs in the image
        if img.dtype != np.uint8:
            img = np.array(img, np.uint8)
        if len(img.shape) > 2:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea)
        for i, cnt in enumerate(contours[:-1]):
            x, y, w, h = cv.boundingRect(cnt)   #c,r,c,r
            contour_area = cv.contourArea(cnt)
            if x == 0 or y == 0 or (x+w - img.shape[1] < margin) or (y+h - img.shape[0] < margin) or (contour_area < min_contour_area):
                cv.drawContours(img, contours, i, (0, 0, 0), -1)
        return img

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
            #if len (st_switches.num_channels)>
            """try :
                brainimg = Slide.read_region((y * self.dfm0, Dims[0][1] - ((x + w) * self.dfm0)), self.blevel, (hb, wb)).convert("BGR")#.convert("RGB")
                brainimg2 = np.array(brainimg)
            except : """
            brainimg = Slide.read_region((y * self.dfm0, Dims[0][1] - ((x + w) * self.dfm0)), self.blevel, (hb, wb))#.convert("BGR")#.convert("RGB")
            brainimg2 = cv.cvtColor(np.array(brainimg),cv.COLOR_BGR2RGB)
            #brainimg2 = cv.cvtColor(np.array(brainimg2),cv.COLOR_BGR2RGB)
            """_=brainimg2[:,:,0]
            brainimg2[:,:,0]=brainimg2[:,:,2]
            brainimg2[:,:,2]=_"""
            #brainimg2=cv.cvtColor(brainimg2,cv.COLOR_RGB2BGR)
            print (brainimg2.shape , "brainimgbrainimgbrainimgbrainimg",brainimg2[brainimg2>20].shape)
            if not st_switches.rotate_flag:
    
                section_blevel = cv.rotate(brainimg2, cv.ROTATE_90_CLOCKWISE)
            else : 
                section_blevel = brainimg2
            if st_switches.color_switch_on:
                section_blevel2 = section_blevel.copy()
                section_blevel2[:,:,1]= section_blevel[:,:,2]
                section_blevel2[:,:,2]= section_blevel[:,:,1]
                section_blevel = section_blevel2
            section_blevel_eq = equalize_img(section_blevel)
            """ try :
                braina = Slide.read_region((y * self.dfm0, Dims[0][1] - ((x + w) * self.dfm0)), self.alevel, (ha, wa)).convert("BGR")#.convert("RGB")
                braina_dark = np.array(braina)
                #braina_dark = cv.cvtColor(np.array(braina),cv.COLOR_RGB2BGR)
            except : """
            braina = Slide.read_region((y * self.dfm0, Dims[0][1] - ((x + w) * self.dfm0)), self.alevel, (ha, wa))#.convert("BGR")#.convert("RGB")
            
            braina_dark = cv.cvtColor(np.array(braina),cv.COLOR_BGR2RGB)
            #braina_dark = cv.cvtColor(np.array(braina_dark),cv.COLOR_BGR2RGB)
            """_=braina_dark[:,:,0]
            braina_dark[:,:,0]=braina_dark[:,:,2]
            braina_dark[:,:,2]=_"""
            #braina_dark=cv.cvtColor(braina_dark,cv.COLOR_RGB2BGR)
            if st_switches.rotate_flag:

                braina_rot = cv.rotate(braina_dark, cv.ROTATE_90_CLOCKWISE)
                section_alevel = braina_rot
            else : 
                section_alevel = braina_dark
            print (section_alevel.shape,"shap[eeeeeeeeee]")
            if st_switches.color_switch_on:
                section_alevel2 = section_alevel.copy()
                section_alevel2[:,:,1]= section_alevel[:,:,2]
                section_alevel2[:,:,2]= section_alevel[:,:,1]
                section_alevel = section_alevel2
            print (section_alevel.shape,"shap[eeeeeeeeee]")
            section_alevel_eq = equalize_img(section_alevel)
            
            #####
            """blevelstack=[]
            for index in range(section_blevel.shape[2]):
                #channel_name = self.channel_types[channel]
                #blevel_channel = self.czi.czi_section_img(self.slidepath, brnum0, num_sections, self.blevel, [channel], rect=None)
                blevel_channel = section_blevel[..., index]
                if st_switches.gammas[index]=="default":
                    gamma_corrected_image = imgprc.gamma_correction(blevel_channel)
                else:
                    gamma_corrected_image = imgprc.gamma_correction(blevel_channel, st_switches.gammas[index])
                #sharpened_image = imgprc.apply_sharpening(gamma_corrected_image)
                #sharpened_image=cv.bitwise_and(gamma_corrected_image, gamma_corrected_image, mask = blevel_mask)

                blevelstack.append(gamma_corrected_image)
                if st_switches.rotate_flag:
                    cv.imwrite(os.path.join(self.section_savepath, f"blevel_{self.channel_types[channel]}.png"), cv.rotate(sharpened_image, cv.ROTATE_90_CLOCKWISE))
                    #cv.imwrite(os.path.join(self.section_savepath, f"blevel_{self.channel_types[channel]}.png"), cv.rotate(blevel_channel, cv.ROTATE_90_CLOCKWISE))

            blevel_b, blevel_g, blevel_r =cv.split(section_blevel)# blevelstack#cv.split(section_blevel)
            section_blevel_eq = czi_channel_regulator(np.dstack(blevelstack))"""
            #section_alevel_eq = equalize_img(section_alevel)
            blevel_b, blevel_g, blevel_r = cv.split(section_blevel)
            cv.imwrite(os.path.join(self.section_savepath,"alevel.png"), section_alevel)
            cv.imwrite(os.path.join(self.section_savepath,"blevel.png"), section_blevel)
            cv.imwrite(os.path.join(self.section_savepath,"alevel_eq.png"), section_alevel_eq)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_eq.png"), section_blevel_eq)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_2.png"), blevel_b)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_1.png"), blevel_g)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_0.png"), blevel_r)
        
        elif self.slideformat == "czi":
            num_sections = len(brainboundcoords)
            
            section_alevel = self.czi.czi_section_img(self.slidepath, brnum0, num_sections, self.alevel, st_switches.num_channels, rect=None)
            section_blevel = self.czi.czi_section_img(self.slidepath, brnum0, num_sections, self.blevel, st_switches.num_channels, rect=None)
            if st_switches.rotate_flag:
                section_alevel=cv.rotate(section_alevel, cv.ROTATE_90_CLOCKWISE)
                section_blevel=cv.rotate(section_blevel, cv.ROTATE_90_CLOCKWISE)
            section_alevel = czi_channel_regulator(section_alevel)
            section_blevel = czi_channel_regulator(section_blevel)  #st_switches.num_channels)

            section_alevel_eq = histogram_equalization(section_alevel)
            section_blevel_eq = histogram_equalization(section_blevel) 
            section_alevel_eq_gr0 = cv.cvtColor(section_alevel_eq, cv.COLOR_BGR2GRAY)
            section_alevel_eq_gr = cv.convertScaleAbs(section_alevel_eq_gr0, alpha=(255.0/65535.0))
            section_blevel_eq_gr0=cv.cvtColor(section_blevel_eq, cv.COLOR_BGR2GRAY)
            section_blevel_eq_gr=cv.convertScaleAbs(section_blevel_eq_gr0, alpha=(255.0/65535.0))
            _, alevel_mask = cv.threshold(section_alevel_eq_gr, ALEVEL_MASK_THRESH, 255, cv.THRESH_BINARY)
            
            _, blevel_mask = cv.threshold(section_blevel_eq_gr, BLEVEL_MASK_THRESH, 255, cv.THRESH_BINARY)
            cv.imwrite (os.path.join(self.section_savepath,"blevel_test.png"),blevel_mask)
            alevel_mask_fixed = self.remove_edge_blob(alevel_mask, 20000)
            blevel_mask_fixed = self.remove_edge_blob(blevel_mask, 20000*(2**(self.alevel-self.blevel)))
            cv.imwrite(os.path.join(self.section_savepath,"alevel_mask_fixed.png"), alevel_mask_fixed)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_mask_fixed.png"), blevel_mask_fixed)
            # alevel_mask_fixed =cv.morphologyEx(alevel_mask_fixed, cv.MORPH_CLOSE, kernel2)
            section_alevel_eq = cv.bitwise_and(section_alevel_eq, section_alevel_eq, mask = alevel_mask_fixed)
            
            if CH_O:
                section_alevel_eq[:,:,CH_O-1] = 0
            """if st_switches.rotate_flag:
                pool.apply_async(cv.imwrite, (os.path.join(self.section_savepath,"alevel.png"), cv.rotate(section_alevel, cv.ROTATE_90_CLOCKWISE)))
                pool.apply_async(cv.imwrite, (os.path.join(self.section_savepath,"blevel.png"), cv.rotate(section_blevel, cv.ROTATE_90_CLOCKWISE)))
                pool.apply_async(cv.imwrite, (os.path.join(self.section_savepath,"alevel_eq.png"), cv.rotate(section_alevel_eq, cv.ROTATE_90_CLOCKWISE)))
            else:"""
            pool.apply_async(cv.imwrite,(os.path.join(self.section_savepath,"alevel.png"), section_alevel))
            pool.apply_async(cv.imwrite,(os.path.join(self.section_savepath,"blevel.png"), section_blevel))
            pool.apply_async(cv.imwrite,(os.path.join(self.section_savepath,"alevel_eq.png"), section_alevel_eq))

            blevelstack=[]

            for index,channel in enumerate(st_switches.num_channels):
                #channel_name = self.channel_types[channel]
                #blevel_channel = self.czi.czi_section_img(self.slidepath, brnum0, num_sections, self.blevel, [channel], rect=None)
                blevel_channel = section_blevel[..., index]

                if st_switches.gammas[index]=="default":
                    gamma_corrected_image = imgprc.gamma_correction(blevel_channel)
                else:
                    gamma_corrected_image = imgprc.gamma_correction(blevel_channel, st_switches.gammas[index])
                
                #sharpened_image = imgprc.apply_sharpening(gamma_corrected_image)

                sharpened_image=cv.bitwise_and(gamma_corrected_image, gamma_corrected_image, mask = blevel_mask)

                blevelstack.append(sharpened_image)
                """if st_switches.rotate_flag:
                    cv.imwrite(os.path.join(self.section_savepath, f"blevel_{self.channel_types[channel]}.png"), cv.rotate(sharpened_image, cv.ROTATE_90_CLOCKWISE))
                    #cv.imwrite(os.path.join(self.section_savepath, f"blevel_{self.channel_types[channel]}.png"), cv.rotate(blevel_channel, cv.ROTATE_90_CLOCKWISE))"""
                #else : 
                cv.imwrite(os.path.join(self.section_savepath, f"blevel_{self.channel_types[channel]}.png"), sharpened_image)
                #cv.imwrite(os.path.join(self.section_savepath, f"blevel_{self.channel_types[channel]}.png"), blevel_channel)
            
            section_blevel_eq = czi_channel_regulator(np.dstack(blevelstack))

            """if st_switches.rotate_flag:
                cv.imwrite (os.path.join(self.section_savepath,"blevel_eq.png"), cv.rotate(section_blevel_eq, cv.ROTATE_90_CLOCKWISE))
            else :"""
            cv.imwrite(os.path.join(self.section_savepath,"blevel_eq.png"), section_blevel_eq )

        blob_detection_file_name = os.path.join(self.section_savepath,"blevel_eq.png")
        tissue_lm_detection_filename = os.path.join(self.section_savepath,"alevel_eq.png")
        
        time4=time.time()

        
        pool.close()
        pool.join()
        del (pool)
        print(f"Getting Section Images took {time4-time3}")
        return blob_detection_file_name, tissue_lm_detection_filename
    
    


    def funcBlobDetection(self, brnum, blobs_parameters):
        """ Returns blobs_log_r, blobs_log_g, colocalized_blobs:: list of blob coords (r,c) before 
        adding the blobs added/removed manually by user
        Saves blobs_locs_r, blobs_locs_g, blob_locs_co as numpy.array
        this also include blob coords (r,c) before adding the blobs added/removed manually by user
        Saves  r_params_for_save, g_params_for_save as npy
        """
        timer1=time.time()

        tempMARGIN = 50  # temporary margin just to avoid the borders when applying thresh, adding the margin is reversed in the parameter brain_mask_eroded

        # brain_blevel = cv.imread(os.path.join(self.section_savepath, 'blevel_eq.png'))#,cv.COLOR_BGR2RGB)
        # brainimgtemp_gray = cv.cvtColor(brain_blevel, cv.COLOR_BGR2GRAY)
        # _, brain_mask = cv.threshold(brainimgtemp_gray, BLEVEL_MASK_THRESH, 255, cv.THRESH_BINARY)
        # cv.imwrite(os.path.join(self.section_savepath, 'brain_mask.jpg'), brain_mask)
        czi_images={}
        params={}

        if self.slideformat == "mrxs":
            brain_blevel = cv.imread(os.path.join(self.section_savepath, 'blevel_eq.png'))#,cv.COLOR_BGR2RGB)
            if st_switches.Bright_field:
                print ("herererrrrr")
                for j in range(brain_blevel.shape[2]):
                    brain_blevel[...,j]=(-1)*brain_blevel[...,j]
                    brain_blevel[...,j]=brain_blevel[...,j]-np.min(brain_blevel[...,j])

            brainimgtemp_gray = cv.cvtColor(brain_blevel, cv.COLOR_BGR2GRAY)
            _, brain_mask = cv.threshold(brainimgtemp_gray, BLEVEL_MASK_THRESH, 255, cv.THRESH_BINARY)
            cv.imwrite(os.path.join(self.section_savepath, 'blevel_mask_fixed.png'), brain_mask)
            kernel1 = np.ones((11,11), np.uint8)
            kernel2 = np.ones((27, 27), np.uint8)
            img_channel_0 = cv.imread(os.path.join(self.section_savepath, 'blevel_0.png'), 0)
            img_channel_1 = cv.imread(os.path.join(self.section_savepath, 'blevel_1.png'), 0)
            img_channel_2 = cv.imread(os.path.join(self.section_savepath, 'blevel_2.png'), 0)
            czi_images[0]=img_channel_0
            czi_images[1]=img_channel_1
            czi_images[2]=img_channel_2
            if st_switches.Bright_field:
                for j in range(len(czi_images)):
                    czi_images[j]=(-1)*czi_images[j]
                    czi_images[j]=czi_images[j]-np.min(czi_images[j])



        elif self.slideformat == "czi":

            kernel1 = np.ones((11,11), np.uint8)
            kernel2 = np.ones((11, 11), np.uint8)
            for index,channel in enumerate(st_switches.num_channels):
                name=self.channel_types[channel]
                czi_images[index]=cv.imread(os.path.join(self.section_savepath, f'blevel_{name}.png'), 0)
            #img_channel_r = cv.imread(os.path.join(self.section_savepath, 'blevel_2.png'), 0)
            #img_channel_g = cv.imread(os.path.join(self.section_savepath, 'blevel_1.png'), 0)

        # brain_mask_temp = cv.copyMakeBorder(brain_mask_edge_removed, tempMARGIN, tempMARGIN, tempMARGIN, tempMARGIN, cv.BORDER_CONSTANT, value=(0, 0, 0))
        brain_mask_temp = cv.imread(os.path.join(self.section_savepath, 'blevel_mask_fixed.png'), cv.IMREAD_UNCHANGED)
        print (os.path.join(self.section_savepath, 'blevel_mask_fixed.png'))
        print (brain_mask_temp)
        closing = cv.morphologyEx(brain_mask_temp, cv.MORPH_CLOSE, kernel2)
        # cv.imwrite(os.path.join(self.section_savepath, 'brain_mask_closed.jpg'), closing)
        brain_mask_eroded_uncut = cv.erode(closing, kernel1, iterations=3)
        # cv.imwrite(os.path.join(self.section_savepath, 'brain_mask_eroded.jpg'), brain_mask_eroded_uncut)
        # brain_mask_eroded -= brain_mask_eroded_uncut[tempMARGIN:-tempMARGIN, tempMARGIN:-tempMARGIN]
        brain_mask_eroded = brain_mask_eroded_uncut
        cv.imwrite(os.path.join(self.section_savepath, 'brain_mask_eroded_cut.jpg'), brain_mask_eroded)


        ### Parameters
        params[0]=blobs_parameters["c0_blob_type"]
        params[1]=blobs_parameters["c1_blob_type"]
        for i in range(2,len (st_switches.num_channels)):
            params[i]=st_switches.type_channels[i]

        #red_blob_type = blobs_parameters["c0_blob_type"]
        #green_blob_type = blobs_parameters["c0_blob_type"]

        sharpening_kernel = np.array([[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]])


        blobs_parameters_dict_to_save = {}
        ######### Red blobs detection
        
        self.blob_logs=[]
        
        for index,blob_type in params.items():
            print ("currently on ", index, blob_type)
            pool=Pool()
            #print("blobs_parameters  ", blobs_parameters)
            if blob_type == "Rabies" or blob_type == "r" :
                if index in (0,1):
                    minsize = blobs_parameters[f'c{str(index)}_blob_min_size']
                    red_blobs_thresh = blobs_parameters[f"c{str(index)}_blob_rabies_thresh"]
                else :
                    minsize, red_blobs_thresh = st_switches.params_rabies[index]     #blobs_parameters['c0_blob_min_size']
                    #red_blobs_thresh = blobs_parameters["red_blob_thresh"]
                
                self.blob_logs.append( self.rabies_detection(czi_images[index], red_blobs_thresh, minsize, brain_mask_eroded))
                #r_params_for_save = np.array([minsize,red_blobs_thresh])
                blobs_parameters_dict_to_save[index] = [minsize, red_blobs_thresh]
                #np.save(os.path.join(section_savepath, 'blobparams_r.npy'), r_params_for_save)

            if blob_type == "MoG" or blob_type == "m":
                min_corr = blobs_parameters['red_blob_correlation']
                stride = blobs_parameters['red_blob_stride']
                self.blobs_log_r = MoG_detection(img_channel_r, min_corr, stride, brain_mask_eroded)
            
            elif blob_type == "cFos" or blob_type=="c" :
                if index in (0,1):
                    minsigma = blobs_parameters[f'c{str(index)}_blob_min_sigma']
                    maxsigma = blobs_parameters[f'c{str(index)}_blob_max_sigma']
                    r_thresh = blobs_parameters[f'c{str(index)}_blob_cfos_thresh1']
                    red_blobs_thresh = blobs_parameters[f'c{str(index)}_blob_cfos_thresh2'] /100
                else:
                    minsigma, maxsigma, r_thresh, red_blobs_thresh = st_switches.params_cfos[index]
                    red_blobs_thresh = red_blobs_thresh /100
                #print(index, minsigma, maxsigma, r_thresh, red_blobs_thresh)
                numsigma = 10
                cv.imwrite(os.path.join(self.section_savepath, f"zz_c{str(index)}_czi_images.png"), czi_images[index])
                img_channel_r_temp = czi_images[index]
                # img_channel_r_temp = cv.convertScaleAbs(czi_images[index], alpha=ALPHA, beta=0)
                # cv.imwrite(os.path.join(self.section_savepath, f"zz_c{str(index)}_enhanced.png"), img_channel_r_temp)
                _, ch_thresh = cv.threshold(img_channel_r_temp, r_thresh, 255, cv.THRESH_BINARY)
                img_channel_r = cv.bitwise_and(img_channel_r_temp, img_channel_r_temp, mask = ch_thresh)
                cv.imwrite(os.path.join(self.section_savepath, f"zz_c{str(index)}_blobmask.png"), ch_thresh)
                cv.imwrite(os.path.join(self.section_savepath, f"zz_c{str(index)}_masked.png"), img_channel_r)
                #img_channel_r = cv.filter2D(czi_images[index], -1, sharpening_kernel)
                #cv.imwrite(os.path.join(self.section_savepath, f"zz_c{str(index)}_sharpened.png"), img_channel_r)
                patches,rx,cx,rstep,cstep= pool_cell_detection(img_channel_r, brain_mask_eroded, minsigma, maxsigma, numsigma, red_blobs_thresh)
                #self.blob_logs.append( pool_cell_detection(czi_images[index], brain_mask_eroded, minsigma, maxsigma, numsigma, red_blobs_thresh, "red_cells"))

                processes=[]
                outputs=[]
                for i in patches[:10]:
                    processes.append(pool.apply_async(cfos_detection,[i]))
                for i in processes:
                    outputs.append(i.get())
                    #OUTPUTS.OUTPUT.append(result.get()
                
                """pool.close()
                pool.join()"""
                self.blob_logs.append(calc_new_coords(outputs,rx,cx,rstep,cstep))
                
                blobs_parameters_dict_to_save[index] = [minsigma, maxsigma, numsigma, red_blobs_thresh]
                     
            pool.close()
            pool.join()
            del(pool)
        match_counts,blob_locs_co = calculate_colocalized_blobs(self.blob_logs)#self.blobs_log_r, self.blobs_log_g)
        # matchcount, blob_locs_co = calculate_colocalized_blobs(self.blob_logs)#self.blobs_log_r, self.blobs_log_g)
        
        
        #number_of_blobs_g = len(self.blobs_log_g)
        #number_of_blobs_r = len(self.blobs_log_r)
        number_of_blobs=[]
        for i in self.blob_logs:
            number_of_blobs.append(len(i))
        saved_data_pickle['blobs_parameters'] = blobs_parameters_dict_to_save

        screenimg_path = os.path.join(self.section_savepath, 'blevel_eq.png')

        for index in czi_images:
            img=czi_images[index].copy()
            for j in self.blob_logs[index]:
                img=cv.circle(img,(j[1],j[0]),st_switches.blob_sizes[index],(255,255,255))
            cv.imwrite(os.path.join(self.section_savepath,f"{self.channel_types[st_switches.num_channels[index]]}_blobs.png"),img)

        return number_of_blobs, match_counts, screenimg_path, self.blob_logs, blob_locs_co
        #return number_of_blobs_r, number_of_blobs_g, matchcount, screenimg_path, self.blobs_log_r, self.blobs_log_g, blob_locs_co
    

    #def funcAnalysis(self,atlasnum, brnum, atlas_prepath, red_blobs_modified, green_blobs_modified, colocalized_blobs_coords):
    def funcAnalysis(self,atlasnum, brnum, atlas_prepath, blobs_coords, colocalized_blobs_coords,blobs_parameters):
        """ Inputs red/green_blobs_modified as a list of blob coords (c, r)
        these include detected blob coords after user modification
        red/green_blobs_modified coords are in blevel
        Intermediate variable blob_locs_r/g np array  [r,c]
        """

        ### Parameters
        params={}
        params[0]=blobs_parameters["c0_blob_type"]
        params[1]=blobs_parameters["c1_blob_type"]
        if len(st_switches.type_channels)>2:
            for j in range(2,len(st_switches.type_channels)+1):
                params[j]=st_switches.type_channels[j]


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

        #num_red_blobs = len(red_blobs_modified)
        #num_gr_blobs = len(green_blobs_modified)
        atlas_width = mappedatlas_labled_showimg.shape[1]

        
        atlas_pil = Image.open(unlabeled_atlas_filepath).convert("RGB")#.convert('RGB')#ERROR
        atlas_colors = Counter(atlas_pil.getdata())
        #red_blobs_coords = red_blobs_modified #(c,r
        saved_data_pickle['blobs_coords_registered'] = blobs_coords
        increamenter=0
        reportfile = open(os.path.join(self.section_savepath, "reportfile.txt"), 'w')
        for i in range(len(st_switches.num_channels)):
            blobs = blobs_coords[i]
            coloc=deepcopy(blobs)
            for coord in range(len(coloc)):
                #try:
                if coloc[coord][0]<0 or coloc[coord][1]<0:
                    blobs.remove((coloc[coord][0],coloc[coord][1]))
        for i in range(len(st_switches.num_channels)):
            redpointcolors = []
            redpointtags = []
            for point in blobs_coords[i]:
                co2, ro2 = point  # Level 3  c, r = xo1, yo1
                try:
                    bb,gg,rr = mappedatlas_detection[ro2, co2]
                    pointcolor = (bb,gg,rr) 
                    pointcolor_rgb = (bb,gg,rr) #(rr,gg,bb) 
                except:
                    pointcolor = (0,0,0) 
                    pointcolor_rgb = (0,0,0)
                cv.circle(mappedatlas_unlabled_showimg, (co2, ro2), 4, BlobColor_[i], -1)
                cv.circle(mappedatlas_unlabled_showimg, (co2, ro2), 4, (0, 0, 0), 1)
                cv.circle(mappedatlas_labled_showimg, (co2, ro2), 4, BlobColor_[i], -1)
                cv.circle(mappedatlas_labled_showimg, (co2, ro2), 4, (0, 0, 0), 1)
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
            cv.imwrite(os.path.join(self.section_savepath, "zz_mappedatlas_unlabled_showimg.jpg"), mappedatlas_unlabled_showimg)
            reportfile.write(f"{len(blobs_coords[i])} blobs in {self.channel_types[st_switches.num_channels[i]]}({i})")#'{} Red Blobs in:\n'.format(len(blobs_coords[i])))
            reportfile.write('\n')
            dict_red  = {'type': f"{self.channel_types[st_switches.num_channels[i]]}({i})", 'Total': str(len(blobs_coords[i]))}
            dict_regs_n_colors={}
            for colortag, count in segcountedr.items():
                pointcolor = self.Rgb_Color_list[colortag[0]]
                if colortag[1]==1:
                    label = Regions_n_colors_list[colortag[0]][-3] + "_L"
                elif colortag[1]==2:
                    label = Regions_n_colors_list[colortag[0]][-3] + "_R"       
                dict_regs_n_colors[label] = [pointcolor, count]
                dict_red[label] = count
                reportfile.write(label + '\t' + str(count) + '\n')
                
            reportfile.write('\n')
            row_red = dict(list(dict_base.items()) + list(dict_red.items())+list({'id':i}.items()))
            for regname in regs_per_section[int(atlasnum)]:
                regname_l = regname + "_L"
                regname_r = regname + "_R"
                if regname_l not in row_red:
                    row_red[regname_l]=0
                if regname_r not in row_red:
                    row_red[regname_r]=0
            Report_subdf = Report_subdf._append(row_red,ignore_index=True)
            if "c" in params[i].lower():#st_switches.type_channels[i].lower() or "c" in params[i].lower():
                
                dict_density = {'type': f'Density_{self.channel_types[st_switches.num_channels[i]]}({i})', 'Total': '__'}
                dict_area = {'type': f'Area_{self.channel_types[st_switches.num_channels[i]]}({i})', 'Total': '__'}

                for regname, value  in dict_regs_n_colors.items():

                    if 'not detected' not in regname.lower():
                        region_cfos_count = value[1] 
                        #if st_switches.atlas
                        region_area = atlas_colors[value[0]]
                        region_density =  region_cfos_count / region_area 
                        dict_area[regname] = region_area 
                        dict_density[regname] = region_density

                row_area = dict(list(dict_base.items()) + list(dict_area.items())+list({'id':i+len(st_switches.num_channels)+increamenter}.items()))
                Report_subdf = Report_subdf._append(row_area, ignore_index=True)

                row_density = dict(list(dict_base.items()) + list(dict_density.items())+list({'id':i+len(st_switches.num_channels)+increamenter+1}.items()))
                Report_subdf = Report_subdf._append(row_density, ignore_index=True)
                increamenter+=1
            #blobs_coords_registered = blobs_coords#{'red': red_blobs_modified, 'green': green_blobs_modified, 'coloc': colocalized_blobs_coords}
            



        for i in range(len(colocalized_blobs_coords)):
            colocalized_blobs = colocalized_blobs_coords[i][0]
            coloc=deepcopy(colocalized_blobs)
            for coord in range(len(coloc)):
                #try:
                if coloc[coord][0]<0 or coloc[coord][1]<0:
                    colocalized_blobs.remove((coloc[coord][0],coloc[coord][1]))
                
                #except IndexError:
                #    break
                

        for i in range(len(colocalized_blobs_coords)):
            colocalized_blobs = colocalized_blobs_coords[i][0]
            matchcount = len(colocalized_blobs)
            #print (matchcount,"countttt")
            #blob_colocs = np.array(colocalized_blobs)
            reportfile.write('\n')
            reportfile.write('{} Co-localization in:\n'.format(matchcount))
            matchpointtags = []
            if matchcount > 0:
                for point in colocalized_blobs:
                    try:
                        bb,gg,rr = mappedatlas_detection[ro2, co2]
                        pointcolor = (bb,gg,rr) 
                        pointcolor_rgb = (bb,gg,rr) #(rr,gg,bb) 
                    except:
                        pointcolor = (0,0,0) 
                        pointcolor_rgb = (0,0,0) 
                    cv.circle(mappedatlas_unlabled_showimg, (co2, ro2), 5, (0, 255, 255), -1)
                    cv.circle(mappedatlas_unlabled_showimg, (co2, ro2), 5, (0, 150, 150), 1)
                    cv.circle(mappedatlas_labled_showimg, (co2, ro2), 4, (0, 255, 255), -1)
                    cv.circle(mappedatlas_labled_showimg, (co2, ro2), 4, (0, 0, 0), 1)
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
            dict_co = {'type': f'CoLoc{i}', 'Total': matchcount}

            if len(matchpointtags)>0:
                for colortag, count in segcountedm.items():
                    pointcolor = self.Rgb_Color_list[colortag[0]]
                    if colortag[1]==1:
                        label = Regions_n_colors_list[colortag[0]][-3] + "_L"
                    elif colortag[1]==2:
                        label = Regions_n_colors_list[colortag[0]][-3] + "_R"      

                    reportfile.write(label + '\t' + str(count) + '\n')
                    dict_co[label] = count
            row_coloc = dict(list(dict_base.items()) + list(dict_co.items())+list({'id':i+len(st_switches.num_channels)+increamenter+2}.items()))
            for regname in regs_per_section[int(atlasnum)]:
                regname_l = regname + "_L"
                regname_r = regname + "_R"
                if regname_l not in row_coloc:
                    row_coloc[regname_l]=0
                if regname_r not in row_coloc:
                    row_coloc[regname_r]=0


            temp=Report_subdf
            Report_subdf = Report_subdf._append(row_coloc, ignore_index=True)



        try: 
            if 'id' in self.Report_df:
                self.Report_df = self.Report_df.set_index('id')
        except : 
            pass
        Report_subdf = Report_subdf.set_index('id')


        if self.report_flag:
            

            if brnum in self.Report_df["Section"].values:
                ms=self.Report_df.index[self.Report_df['Section'] == brnum]
                #Report_subdf=Report_subdf.set_index(ms)

                tmp1=self.Report_df.loc[0:min(ms)-1]
                tmp2=self.Report_df.loc[max(ms)+1:]
                self.Report_df=pd.concat([tmp1,Report_subdf,tmp2],axis=0)
                self.Report_df=self.Report_df.drop_duplicates(["type","Section"])
                #self.Report_df.loc[ms]=Report_subdf.loc[:]
        
                #self.Report_df = self.Report_df[self.Report_df['Section'] != brnum]
                #the line above removes the row with same section number
            else :
                print("section does not exist in report,appending")
                for i in range(Report_subdf.shape[0]):
                    self.Report_df=self.Report_df._append(Report_subdf.iloc[i])
  
        else :
            self.report_flag=True
            self.Report_df=Report_subdf

        try:
            writer = pd.ExcelWriter(os.path.join(savepath, f'Report_{self.slidename}.xlsx'), engine='openpyxl')
            self.Report_df.to_excel(writer, sheet_name='Sheet 1', index=False)
            print("report created")
            writer.close()
            
            writer = pd.ExcelWriter(os.path.join(self.section_savepath, f'Report_{self.slidename}_S{brnum}.xlsx'), engine='openpyxl')
            Report_subdf.to_excel(writer, sheet_name='Sheet 1', index=False)
            writer.close()
        except Exception as E:
            print(E)
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
        cv.imwrite(os.path.join(self.section_savepath, 'zz_rabies_mask.jpg'), mask_eroded)
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

            cv.imwrite(os.path.join(self.section_savepath,"blevel_0.png"), blevel_b)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_1.png"), blevel_g)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_2.png"), blevel_r)
        elif self.slideformat == "czi":
            for index,channel in enumerate(st_switches.num_channels):
                blevel_ch0=cv.flip(cv.imread(os.path.join(self.section_savepath, f"blevel_{self.channel_types[channel]}.png")), 1)
                cv.imwrite(os.path.join(self.section_savepath, f"blevel_{self.channel_types[channel]}.png"),blevel_ch0)
            """blevel_ch0 = cv.flip(cv.imread(os.path.join(self.section_savepath,"blevel_0.png")), 1)
            blevel_ch1 = cv.flip(cv.imread(os.path.join(self.section_savepath,"blevel_1.png")), 1)
            blevel_ch2 = cv.flip(cv.imread(os.path.join(self.section_savepath,"blevel_2.png")), 1)

            cv.imwrite(os.path.join(self.section_savepath,"blevel_0.png"), blevel_ch0)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_1.png"), blevel_ch1)
            cv.imwrite(os.path.join(self.section_savepath,"blevel_2.png"), blevel_ch2)
            """
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


    def calculate_fp_fn_blobs(self,final_blobs,detected_blobs):
        #global saved_data_pickle
        blobs_fp_fn = {}
        reportfile = open(os.path.join(self.section_savepath, "reportfile_fpfn.txt"), 'w')
        for i,j in enumerate(st_switches.num_channels):
            blobs_channel=final_blobs[i]
            blobs_detected=detected_blobs[i]
            blobs_detected = [(sub[1], sub[0]) for sub in blobs_detected] 
            b1=f'channel_{self.channel_types[st_switches.num_channels[i]]}({i})_fn'
            b2=f'channel_{self.channel_types[st_switches.num_channels[i]]}({i})_fp'
            blobs_fp_fn[b1] = [item for item in blobs_channel if item not in blobs_detected] 
            blobs_fp_fn[b2] = [item for item in blobs_detected  if item not in blobs_channel] 
            reportfile.write(f'\n {i} FP: {len(blobs_fp_fn[b2])} and FN: {len(blobs_fp_fn[b1])}')

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



