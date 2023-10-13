import slideio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET


def histogram_equalization(img_in):
    b,g,r = cv2.split(img_in)
    clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(5,5))
    equ_b = clahe.apply(b)
    equ_g = clahe.apply(g)
    equ_r = clahe.apply(r)
    img_out = cv2.merge((equ_b, equ_g, equ_r))
    return img_out

def czi_channel_regulator(image):
    if image.shape[2]==2:
        chan3 = np.zeros_like(image[:,:,0])
        image = np.dstack((image, chan3))
    if image.shape[2]>=4:
        image = image[:,:,0:3]
    return image

def get_images_size(path, dfm0):
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
    channels = root[0][4][3][11][0]
    allchannels = channels.findall('Channel')
    num_channels = len(allchannels)
    channel_types = []
    for channel in allchannels:
        channel_types.append(channel.attrib['Name'])
    return num_channels, channel_types
