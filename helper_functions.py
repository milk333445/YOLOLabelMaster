import cv2
import numpy as np
import math
import torch
import torch.nn as nn
def DrawText(img, 
             text,
             font=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.5,
             font_thickness=1,
             text_color=(0, 0, 255),
             text_color_bg=(255, 255, 255),
             pos='tl',
             axis=(0, 0)
             ):
    axis = (int(axis[0]), int(axis[1]))
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    try:
        if pos == 'bl':
            cv2.rectangle(img, axis, (axis[0]+text_w, axis[1]-text_h*2), text_color_bg, -1)
            cv2.putText(img, text, (axis[0], int(axis[1] - text_h/2)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        elif pos == 'tl':
            cv2.rectangle(img, axis, (axis[0]+text_w, axis[1]+text_h*2), text_color_bg, thickness=-1)
            cv2.putText(img, text, (axis[0], int(axis[1] + text_h*3/2)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        elif pos == 'tr':
            cv2.rectangle(img, axis, (axis[0]-text_w, axis[1]+text_h*2), text_color_bg, thickness=-1)
            cv2.putText(img, text, (axis[0]-text_w, int(axis[1] + text_h*3/2)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        elif pos == 'br':
            cv2.rectangle(img, axis, (axis[0]-text_w, axis[1]-text_h*2), text_color_bg, thickness=-1)
            cv2.putText(img, text, (axis[0]-text_w, int(axis[1] - text_h/2)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    except:
        print('position and axis are wrong setting')
        
        
        
