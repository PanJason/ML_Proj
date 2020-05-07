# Do data processing in this file
import cv2
import option
import numpy as np

def dataprocess(input_img):
    padding_size=(3056,3056,3)
    output_img=np.zeros(padding_size)
    w=min(padding_size[0],input_img.shape[0])
    h=min(padding_size[1],input_img.shape[1])
    output_img[0:w,0:h]=input_img[0:w,0:h]

    huidu=160
    output_img=output_img*160/np.mean(input_img)
    return output_img

