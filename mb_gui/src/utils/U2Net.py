from skimage import io, transform, color
import numpy as np
import Switches_Static as st_switches
import Switches_Dynamic as dy_switches
import os
import onnxruntime as ort
import cv2
from PIL import Image

rootpath = dy_switches.get_rootpath()
u2netp_path=os.path.join(rootpath,"mb_gui/models","u2netp.onnx")
session = ort.InferenceSession(u2netp_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name 
def RescaleT(image,target_shape):
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = self.output_size*h/w,self.output_size
    else:
        new_h, new_w = self.output_size,self.output_size*w/h
    new_h, new_w = int(new_h), int(new_w)
    img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
    return img


def RescaleTFunc(image,target_shape=320):
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = target_shape*h/w,target_shape
    else:
        new_h, new_w = target_shape,target_shape*w/h
    new_h, new_w = int(new_h), int(new_w)
    img = transform.resize(image,(target_shape,target_shape),mode='constant')
    return img

def ToTensorLabFunc(image):
    tmpImg = np.zeros((image.shape[0],image.shape[1],3))
    image = image/np.max(image)
    if image.shape[2]==1:
        tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
    else:
        tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
        tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225


    # change the r,g,b to b,r,g from [0,255] to [0,1]
    #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    tmpImg = tmpImg.transpose((2, 0, 1))
    return tmpImg.astype("float32")

def post_process(img,image_orig):

    predict_np = img.squeeze()
    im = Image.fromarray(predict_np*255)#.convert('RGB')
    out_img = im.resize((image_orig.shape[1],image_orig.shape[0]),resample=Image.BILINEAR)
    
    #path="C:\AMBIA2\AMBIA\Processed\\2023_04_02__4829_SAMPLELamin_new set begining_DAB-Split Scenes-02\S1\\test.png"
    #out_img.save(path)
    out_img=np.array(out_img)/255
    out_img[out_img > st_switches.U2NET_THRESHOLD] = 1
    out_img[out_img <= st_switches.U2NET_THRESHOLD] = 0
    out_image_binary=out_img.copy()
    out_img*=255
    return out_img,out_image_binary

def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def background_removal(img):
    img_orig=img.copy()
    img=RescaleTFunc(img,320)
    img=ToTensorLabFunc(img)
    img=np.array([img])
    output = session.run([output_name], {input_name: img})[0]
    output=output[:,0,:,:]
    output = normPRED(output)
    processed_output,processed_output_binary=post_process(output,img_orig)
    return output,processed_output,processed_output_binary