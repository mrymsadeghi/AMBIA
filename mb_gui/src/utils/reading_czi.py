import slideio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET
import Switches_Static as st_switches


CNTR_ENH = st_switches.contrast_enhancement

def histogram_equalization(img_in):
    b,g,r = cv2.split(img_in)
    clahe = cv2.createCLAHE(clipLimit=CNTR_ENH, tileGridSize=(10,10))
    equ_b = clahe.apply(b)
    equ_g = clahe.apply(g)
    equ_r = clahe.apply(r)
    img_out = cv2.merge((equ_b, equ_g, equ_r))
    return img_out
class CZI:
    def __init__(self,path):
        self.slide = slideio.open_slide(path,"CZI")
    def get_images_size(self,path, dfm0):
        
        brainboundcoords = np.array([[0, 0, 0, 0]])
        for i in range(self.slide.num_scenes):
            scene = self.slide.get_scene(i)
            brainboundcoords = np.append(brainboundcoords, [[int(-scene.rect[0]/dfm0), int(scene.rect[1]/dfm0), int(scene.rect[2]/dfm0), int(scene.rect[3]/dfm0)]], axis=0)
        return brainboundcoords
    def czi_section_img(self,path, section_num, num_sections, downsample_num, channels, rect=None):
    
        image_number = num_sections - section_num
        scene = self.slide.get_scene(image_number)
        downrate = 2 ** downsample_num
        s = scene.size
        
        if(rect == None):
            rect = (0,0,s[0],s[1])
        image = scene.read_block(rect, (int(s[0]/downrate),int(s[1]/downrate)), channels)
        return image
    
    def czi_preview(self,slidepath, savepath, downsample):
        #slide = slideio.open_slide(slidepath, "CZI")
        downrate = 2**downsample
        plt.subplots_adjust(wspace=0.1, hspace=0.001)
        plt.figure(figsize=(20, 10), facecolor='black')
        num_col = int(np.ceil(self.slide.num_scenes/2))
        num_scenes_img = self.slide.num_scenes 
        for i in range(num_scenes_img):
            plt.subplot(2, num_col, num_scenes_img-i)
            plt.axis("off")
            scene = self.slide.get_scene(i)
            s = scene.size
            image = scene.read_block((0, 0, s[0], s[1]), (int(s[0]/downrate), int(s[1]/downrate)))
            if image.shape[2]==2:
                chan3 = np.zeros_like(image[:,:,0])
                image = np.dstack((image, chan3))
            if image.shape[2]>=4:
                image = image[:,:,0:3]
            img=(image/256).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_output = histogram_equalization(img)
            plt.imshow(img_output)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(savepath, facecolor='black')

    def get_channel_info(self,slidepath):
        #slide = slideio.open_slide(slidepath, "CZI")
        metadata = self.slide.raw_metadata
        root = ET.fromstring(metadata)
        #root = etree.fromstring(metadata)

        try:
            channels = root[0][4][3][11][0]

        except:
            
            channels = root[0][3][3][11][0]

        allchannels = channels.findall('Channel')


        num_channels = len(allchannels)
        channel_types = []
        for channel in allchannels:
            channel_types.append(channel.attrib['Name'])
        return num_channels, channel_types
    








def czi_channel_regulator(image):
    if image.shape[2]==2:
        chan3 = np.zeros_like(image[:,:,0])
        image = np.dstack((image, chan3))
    if image.shape[2]>=4:
        image = image[:,:,0:3]
    return image



"""def get_images_size(path, dfm0):
    slide = slideio.open_slide(path,"CZI")
    brainboundcoords = np.array([[0, 0, 0, 0]])
    for i in range(slide.num_scenes):
        scene = slide.get_scene(i)
        brainboundcoords = np.append(brainboundcoords, [[int(-scene.rect[0]/dfm0), int(scene.rect[1]/dfm0), int(scene.rect[2]/dfm0), int(scene.rect[3]/dfm0)]], axis=0)
    return brainboundcoords

def czi_section_img(path, section_num, num_sections, downsample_num, channels, rect=None):
    slide = slideio.open_slide(path,"CZI")
    image_number = num_sections - section_num
    scene = slide.get_scene(image_number)
    downrate = 2 ** downsample_num
    s = scene.size
    if(rect == None):
        rect = (0,0,s[0],s[1])
    image = scene.read_block(rect, (int(s[0]/downrate),int(s[1]/downrate)), channels)
    return image


def czi_preview(slidepath, savepath, downsample):
    slide = slideio.open_slide(slidepath, "CZI")
    downrate = 2**downsample
    plt.subplots_adjust(wspace=0.1, hspace=0.001)
    plt.figure(figsize=(20, 10), facecolor='black')
    num_col = int(np.ceil(slide.num_scenes/2))
    num_scenes_img = slide.num_scenes 
    for i in range(num_scenes_img):
        plt.subplot(2, num_col, num_scenes_img-i)
        plt.axis("off")
        scene = slide.get_scene(i)
        s = scene.size
        image = scene.read_block((0, 0, s[0], s[1]), (int(s[0]/downrate), int(s[1]/downrate)))
        if image.shape[2]==2:
            chan3 = np.zeros_like(image[:,:,0])
            image = np.dstack((image, chan3))
        if image.shape[2]>=4:
            image = image[:,:,0:3]
        img=(image/256).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_output = histogram_equalization(img)
        plt.imshow(img_output)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(savepath, facecolor='black')

def get_channel_info(slidepath):
    slide = slideio.open_slide(slidepath, "CZI")
    metadata = slide.raw_metadata
    root = ET.fromstring(metadata)
    try:
        channels = root[0][4][3][11][0]
    except:
        channels = root[0][3][3][11][0]

    allchannels = channels.findall('Channel')
    num_channels = len(allchannels)
    channel_types = []
    for channel in allchannels:
        channel_types.append(channel.attrib['Name'])
    return num_channels, channel_types
"""





"""
path = "C:/Slides/A1_R3_S4/A1_R3_S4.czi"
img_savepath = "C:/Slides/A1_R3_S4/saved/"
#img_savepath2 = "E:/Slides/MB_slides/Enrica_2020_11_10/saved/sectionplt.jpg"
#img_savepath3 = "E:/Slides/MB_slides/Enrica_2020_11_10/saved/sectioncv.png"
#img_savepath4 = "E:/Slides/MB_slides/Enrica_2020_11_10/saved/sectioncv2.png"

#czi_preview(path, img_savepath, 6)
#image1 = czi_section_img(path, 6, 3)
image1 = czi_section_img(path, 6, 1)
image2 = czi_section_img(path, 6, 2)
image3 = czi_section_img(path, 6, 3)
image4 = czi_section_img(path, 6, 4)

#img_b, img_g,img_r = cv2.split(image)

#image = image.astype(np.uint16)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#cv2.imwrite(img_savepath4, image2)
#cv2.imwrite(img_savepath3, image)
cv2.imwrite(os.path.join(img_savepath,"image2.png"), image2)
cv2.imwrite(os.path.join(img_savepath,"image3.png"), image3)
cv2.imwrite(os.path.join(img_savepath,"image4.png"), image4)
cv2.imwrite(os.path.join(img_savepath,"image1.png"), image1)



plt.imshow(image)
plt.imshow(image)
plt.savefig(img_savepath2, facecolor='black')

cv2.imwrite(img_savepath3, image)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""