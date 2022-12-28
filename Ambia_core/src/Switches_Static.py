

## Smart Functions:
auto_registration = False               # If set to True, the toggle switch in the GUI for automatic registration is by default on
section_classifier_on = False           # If set to True, DL-based localization of AMBIA will be activated
color_switch_on = False                 # If set to True, red and green channels of flouroscent image will switch places
segmentation_1_20_on = False            # If set to true, the Group A segmentation model will be loaded and performed if needed
section_QL_on = False                   # If set to true, the QL predictor model will be loaded and performed if needed


## Atlas Type:
atlas_type = "Adult"        # Set to Adult or P56, atlas type you would like to use

q5value = 70

###### Mirax format parameters
# mlevel is the mask detection level
# blevel in the blob detection (cell detection) level
# alevel is the atlas registration level
mrx_mlevel = 6            
mrx_blevel = 3
mrx_alevel = 5
mrx_num_channels = 3

mrx_maskthresh = 10             #threshorld for binarization of mlevel slide image for section detection

###### Czi format parametes
czi_mlevel = 6
czi_blevel = 2
czi_alevel = 3
czi_maskthresh = 0


#Blob_size shown on Neuro Detection step in GUI
blob1_size_red = 26
blob1_size_green = 25
blob1_size_yellow = 27

#Blob_size shown on Finaly Analysis step in the GUI
blob2_size_red = 4
blob2_size_green = 3
blob2_size_yellow = 4

#Scale of pixel:mm
scale_mm = 0.0412597                # size of each pixel in mm, 0.0412597    for 3Dhistech 250 x20 magnification
MARGIN = 100
num_rows = 5                        # Number of rows in the exported excel sheet for each slice

########################################## 
## Registration 
########################################## 
ACC_DF = 1                 #Accelarating factor for ardent registration. Downscales both images by this factor to speed up the registration process
TEMP_DF = 3/2               #3/2Approximate ratio between the source image and the target image
TARG_DF = 1
source_resolution = 100
target_resolution = 100


Delauney_strength = 6      # higher the value less freedom


##########################################
## Paths
##########################################

PROCESSED_PATH = "C:/PyProjects/MouseBrainProject/MB_GUI/Processed" 
SLIDES_PREPATH = "C:/Slides/MB_slides"