
# Number of channels starting with 0
num_channels = [ 0,1, 2]   # To be defined as a list of channels e.g [1,2,3] or [0,1], indexes starts with 0 -- displayed as BGR

type_channels = ["GUI", "GUI", "r"]
#permutes must be in respect to the original channels
coloc_permutation = [(0,1,2)]   #e.g [(2,1),(2,0)]

#type_channels = ["GUI", "GUI", "r"]
#coloc_permutation = [(0,2), (1,2)]   #e.g [(2,1),(2,0)]
blob_sizes = [20, 20, 20]

# Rabies params for each channel, in the same order as  num channesl, a two element tuple for each channel, first element minsize, second threshold
# First two tuples can be left empty so they can be modified in gui
params_rabies = [(), (), (8,150)]    # (thresh, minsize)

params_rabies = [(), (), (8, 150)]    # (minsize, thresh)

# cfos params for each channel, in the same order as num channels, a 4 element tuple per channel, first element min sigma, second maxsigma
# third bg intensity (0-255) fourth cell intensity (0-10)
params_cfos = [(), (), (4, 8, 10, 2)]     # (min sigma, maxsigma, bg intensity, cell intensity)

# Use gamma to increase the contrast of the channels individually. 0.5 to 2.5
gammas = ["default", "default", "default"]


#number of patches for multiprocessing in cFos detection, output is rx*cx
rx= 2
cx= 2

## Smart Functions:
auto_registration = True
section_classifier_on = False
color_switch_on = False
segmentation_1_20_on = False
section_QL_on = False
#++++++++++++++++++++++++++++++++++++++
# Image Processing parameters
rotate_flag = False
gamma_enhancement_flag = True
czi_maskthresh = 10                  # This threshold is used for the section detection on the whole slide image
contrast_enhancement = 12.0           # for alevel image contras enhancement. This parameter controls the contrast enhancement range [0,20], the higher -> the more contrast enhanced
blevel_mask_threshold = 5           # This parameter adjust the threshold for brain_mask
alevel_mask_threshold = 5
channel_to_omit = 0                 # This is an integer value 1,2,3 which indicates which channel should be omited for registration, set to 0 for none
CELL_OVERLAP = 0.5                  # If two detected cfos cells have more than this amount overlap, only one will be counted
alevel_gamma_factor = 1.4

#we need padding for
#  rotation to prevent disformation, set 0 if no padding is needed
rotation_padding = 50

## Atlas Type:
atlas_type = "Adult"        # Adult or P56 or Rat
Bright_field = False
q5value = 70

###### Mirax format parametes
mrx_mlevel = 7  #6
mrx_blevel = 3
mrx_alevel = 5
mrx_num_channels = 3

mrx_maskthresh = 20

###### Czi format parameters
czi_mlevel = 6
czi_blevel = 1
czi_alevel = 4


"""#Blob_size shown on Neuro Detection step
blob1_size_red = 26
blob1_size_green = 25
blob1_size_yellow = 27

#Blob_size shown on Finaly Analysis step
blob2_size_red = 4
blob2_size_green = 3
blob2_size_yellow = 4"""

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