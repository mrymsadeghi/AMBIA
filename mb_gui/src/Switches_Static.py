

## Smart Functions:
auto_registration = True
section_classifier_on = False
color_switch_on = False
segmentation_1_20_on = False
section_QL_on = False

# Image Processing parameters
rotate_flag=False
czi_maskthresh = 10                 # This threshold is used for the section detection on the whole slide image
contrast_enhancement = 12.0         # This parameter controls the contrast enhancement range [0,20], the higher -> the more contrast enhanced
blevel_mask_threshold = 10         # This parameter adjust the threshold for brain_mask
alevel_mask_threshold = 10

## Atlas Type:
atlas_type = "Adult"        # Adult or P56 or Rat

q5value = 70

###### Mirax format parametes
mrx_mlevel = 6
mrx_blevel = 3
mrx_alevel = 5
mrx_num_channels = 3

mrx_maskthresh = 20

###### Czi format parameters
czi_mlevel = 6
czi_blevel = 1
czi_alevel = 4


#Blob_size shown on Neuro Detection step
blob1_size_red = 26
blob1_size_green = 25
blob1_size_yellow = 27

#Blob_size shown on Finaly Analysis step
blob2_size_red = 4
blob2_size_green = 3
blob2_size_yellow = 4

#Scale of pixel:mm
scale_mm = 0.0412597                # 0.0412597    for 3Dhistech 250 x20 magnification
MARGIN = 50
num_rows = 5

########################################## 
## Registration 
########################################## 
ACC_DF = 1                  #Accelarating factor for ardent registration. Downscales both images by this factor to speed up the registration process
TEMP_DF = 1               #Approximate ratio between the source image and the target image
TARG_DF = 1
source_resolution = 100
target_resolution = 100


Delauney_strength = 6      # higher the value less freedom