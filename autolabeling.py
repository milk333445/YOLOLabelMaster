import os
import pandas as pd
from pathlib import Path
import torch
import copy
import cv2
import numpy as np
import sys
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from helper_functions import *
import argparse
import yaml
from autolabel_config import *
import re


def location(x, y, w, h, im_w, im_h, classes):
    # x, y, w, h: 偵測框的左上角座標及寬高
    # im_w, im_h: 影像的寬高
    tem_lst = []
    ln_x = round((x+w/2)/im_w, 6) # 中心點x占整張圖寬的比例
    ln_y = round((y+h/2)/im_h, 6) # 中心點y占整張圖高的比例
    ob_w = round(w/im_w, 6) # 物件寬度占整張圖寬的比例
    ob_h = round(h/im_h, 6) # 物件高度占整張圖高的比例
    tem_lst.append(classes)
    tem_lst.append(ln_x)
    tem_lst.append(ln_y)
    tem_lst.append(ob_w)
    tem_lst.append(ob_h)
    return tem_lst



def handle_mouse_move_normal(x, y, drawing, param):
    tmp_im = copy.deepcopy(param[1])
    expand_image = add_boarder(tmp_im)
    if drawing:
        if len(param[2]) > 0:
            for i in range(len(param[2])):
                cv2.rectangle(expand_image, (param[0][i][0], param[0][i][1]), 
                                (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
                DrawText(expand_image, obj[param[2][i]], font_thickness=2, font_scale=1, pos='tl', axis= (param[0][i][0], param[0][i][1]))
                
                
        cv2.rectangle(expand_image, (param[0][-1][0], param[0][-1][1]), 
                        (x, y), (clr[param[3]]), 2)
        DrawText(expand_image, obj[param[3]], font_thickness=2, font_scale=1, pos='tl', axis= (param[0][-1][0], param[0][-1][1]))
        cv2.circle(expand_image, (x, y), 10, (clr[param[3]]), 2)
    
    else:
        if len(param[2]) > 0:
            for i in range(len(param[2])):
                cv2.rectangle(expand_image, (param[0][i][0], param[0][i][1]), 
                                (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
                DrawText(expand_image, obj[param[2][i]], font_thickness=2, font_scale=1, pos='tl', axis= (param[0][i][0], param[0][i][1]))
                
        cv2.line(expand_image, (x, 0), (x, tmp_im.shape[0]), (0, 0, 0), 1)
        cv2.line(expand_image, (0, y), (tmp_im.shape[1], y), (0, 0, 0), 1)
    
    return expand_image


def handle_mouse_move(x, y, drawing, param):
    tmp_im = copy.deepcopy(param[1])
    expand_image = add_boarder(tmp_im)
    if drawing:
        if len(param[2]) > 0:
            for i in range(len(param[2])):
                cv2.rectangle(expand_image, (param[0][i][0], param[0][i][1]), 
                              (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
                
                
        cv2.rectangle(expand_image, (param[0][-1][0], param[0][-1][1]), 
                      (x, y), (clr[param[3]]), 2)
        cv2.circle(expand_image, (x, y), 10, (clr[param[3]]), 2)
    else:
        if len(param[2]) > 0:
            for i in range(len(param[2])):
                cv2.rectangle(expand_image, (param[0][i][0], param[0][i][1]), 
                              (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
                
        cv2.line(expand_image, (x, 0), (x, tmp_im.shape[0]), (0, 0, 0), 1)
        cv2.line(expand_image, (0, y), (tmp_im.shape[1], y), (0, 0, 0), 1)
    return expand_image

def handle_left_buttom_up(x, y, param):
    tmp_param = copy.deepcopy(param)
    tmp_param[0][-1].append(x)
    tmp_param[0][-1].append(y)
    
    new_objs = param[2] + [param[3]]
    print('Labeling Completed')
    print('Current Label Count: ', len(tmp_param[0]))
    
    if len(tmp_param[0]) < 7:
        reminder_text = f'Reminder: {7-len(tmp_param[0])} more annotations needed.'
        print(f'Reminder: Label count not reached the maximum, you can still label {7-len(tmp_param[0])} more.')
    elif len(tmp_param[0]) == 7:
        reminder_text = 'Reminder: Annotations are full.'
        print('Reminder: Label limit reached.')
    else:
        reminder_text = f'Reminder: Over the limit by {len(tmp_param[0])-7} annotations.'
        print(f'Reminder: Exceeded label limit by {len(tmp_param[0])-7}')
    
    return tmp_param[0], new_objs, reminder_text


def handle_right_buttom_up(param):
    param[0].pop()
    param[2].pop()
    
    print('Previous Step')
    print('Current Label Count: ', len(param[0]))
    
    if len(param[0]) < 7:
        reminder_text = f'Reminder: {7-len(param[0])} more annotations needed.'
        print(f'Reminder: Label count not reached the maximum, you can still label{7-len(param[0])} more.')
    elif len(param[0]) == 7:
        reminder_text = 'Reminder: Annotations are full.'
        print('Reminder: Label limit reached.')
    else:
        reminder_text = f'Reminder: Over the limit by {len(param[0])-7} annotations.'
        print(f'Reminder: Exceeded label limit by {len(param[0])-7}')
        
    tmp_im = copy.deepcopy(param[1])
    expand_image = add_boarder(tmp_im)  
    
    font_thickness = 2
    font_scale = 0.6
    
    text_size, _ = cv2.getTextSize(reminder_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    word = "[Enter]: Save | [R]: Undo | [A]: Prev | [D]: Skip"
    text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_w, text_h = text_size
    DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, tmp_im.shape[0] + text_h * 2))

    for i in range(len(param[2])):
        cv2.rectangle(expand_image, (param[0][i][0], param[0][i][1]), (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
       
    return expand_image


def handle_right_buttom_up_normal(param):
    param[0].pop()
    param[2].pop()
    
    print('Previous Step')
    print('Current Label Count: ', len(param[0]))
    
    if len(param[0]) < 7:
        reminder_text = f'Reminder: {7-len(param[0])} more annotations needed.'
        print(f'Reminder: Label count not reached the maximum, you can still label{7-len(param[0])} more.')
    elif len(param[0]) == 7:
        reminder_text = 'Reminder: Annotations are full.'
        print('Reminder: Label limit reached.')
    else:
        reminder_text = f'Reminder: Over the limit by {len(param[0])-7} annotations.'
        print(f'Reminder: Exceeded label limit by {len(param[0])-7}')
        
    tmp_im = copy.deepcopy(param[1])
    expand_image = add_boarder(tmp_im)  
    
    font_thickness = 2
    font_scale = 0.5
    
    text_size, _ = cv2.getTextSize(reminder_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    word = "[Enter]: Save | [R]: Undo | [A]: Prev | [D]: Skip | [S]: Next obj | [W]: Prev obj"
    text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_w, text_h = text_size
    DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, tmp_im.shape[0] + text_h * 2))

    for i in range(len(param[2])):
        cv2.rectangle(expand_image, (param[0][i][0], param[0][i][1]), (param[0][i][2], param[0][i][3]), (clr[param[2][i]]), 2)
        DrawText(expand_image, obj[param[2][i]], font_thickness=2, font_scale=1, pos='tl', axis= (param[0][i][0], param[0][i][1]))
    return expand_image


def show_xy(event, x, y, flags, param):
    global drawing
    tmp_im = copy.deepcopy(param[1])
    expand_image = add_boarder(tmp_im)
    
    font_thickness=2
    font_scale=0.7
    word = "[Enter]: Save | [R]: Undo | [A]: Prev | [D]: Skip"
    text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_w, text_h = text_size
    DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, tmp_im.shape[0] + text_h * 2))
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True 
        x1, y1 = x, y
        param[0].append([x1, y1])
        
        print('Start Labeling')
        print('Labeling Objects...')
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            expand_image = handle_mouse_move(x, y, drawing, param)
            x1 = param[0][-1][0]
            y1 = param[0][-1][1]
            
            cv2.line(expand_image, (x1, 0), (x1, tmp_im.shape[0]), (0, 0, 0), 1)
            cv2.line(expand_image, (x, 0), (x, tmp_im.shape[0]), (0, 0, 0), 1)
            cv2.line(expand_image, (0, y1), (tmp_im.shape[1], y1), (0, 0, 0), 1)
            cv2.line(expand_image, (0, y), (tmp_im.shape[1], y), (0, 0, 0), 1)
            font_thickness=2
            font_scale=0.6
            word = "[Enter]: Save | [R]: Undo | [A]: Prev | [D]: Skip"
            text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_w, text_h = text_size
            DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, tmp_im.shape[0] + text_h * 2))
            cv2.imshow(param[4], expand_image)
        else:
            expand_image = handle_mouse_move(x, y, drawing, param)
            font_thickness=2
            font_scale=0.6
            word = "[Enter]: Save | [R]: Undo | [A]: Prev | [D]: Skip"
            text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_w, text_h = text_size
            DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, tmp_im.shape[0] + text_h * 2))
            cv2.imshow(param[4], expand_image)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        new_coords, new_objs, reminder_text = handle_left_buttom_up(x, y, param)
        
        # 更新param
        param[0] = new_coords
        param[2] = new_objs
        
        text_size, _ = cv2.getTextSize(reminder_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        DrawText(expand_image, reminder_text, font_thickness=font_thickness, font_scale=font_scale, pos='tl', axis=(0, 0))
        
        if len(new_coords) > 0:
            for i in range(len(new_coords)):
                cv2.rectangle(expand_image, (new_coords[i][0], new_coords[i][1]), 
                              (new_coords[i][2], new_coords[i][3]), (clr[new_objs[i]]), 2)
                
        
        cv2.imshow(param[4], expand_image)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        expand_image = handle_right_buttom_up(param)
        cv2.imshow(param[4], expand_image)


def show_xy_normal_multilabel(event, x, y, flags, param):
    global drawing
    
    tmp_im = copy.deepcopy(param[1])
    expand_image = add_boarder(tmp_im)
    
    font_thickness=2
    font_scale=0.55
    word = "[Enter]: Save | [R]: Undo | [A]: Prev | [D]: Skip | [S]: Next obj | [W]: Prev obj"
    text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_w, text_h = text_size
    DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, tmp_im.shape[0] + text_h * 2))
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True 
        x1, y1 = x, y
        param[0].append([x1, y1])
        
        print('Start Labeling')
        print('Labeling Objects...')
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            expand_image = handle_mouse_move_normal(x, y, drawing, param)
            x1 = param[0][-1][0]
            y1 = param[0][-1][1]
            
            cv2.line(expand_image, (x1, 0), (x1, tmp_im.shape[0]), (0, 0, 0), 1)
            cv2.line(expand_image, (x, 0), (x, tmp_im.shape[0]), (0, 0, 0), 1)
            cv2.line(expand_image, (0, y1), (tmp_im.shape[1], y1), (0, 0, 0), 1)
            cv2.line(expand_image, (0, y), (tmp_im.shape[1], y), (0, 0, 0), 1)
            font_thickness=2
            font_scale=0.5
            word = "[Enter]: Save | [R]: Undo | [A]: Prev | [D]: Skip | [S]: Next obj | [W]: Prev obj"
            text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_w, text_h = text_size
            DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, tmp_im.shape[0] + text_h * 2))
            cv2.imshow(param[4], expand_image)
        else:
            expand_image = handle_mouse_move_normal(x, y, drawing, param)
            font_thickness=2
            font_scale=0.5
            word = "[Enter]: Save | [R]: Undo | [A]: Prev | [D]: Skip | [S]: Next obj | [W]: Prev obj"
            text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_w, text_h = text_size
            DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, tmp_im.shape[0] + text_h * 2))
            cv2.imshow(param[4], expand_image)
        
        
        
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        new_coords, new_objs, reminder_text = handle_left_buttom_up(x, y, param)
        
        # 更新param
        param[0] = new_coords
        param[2] = new_objs
        reminder_text = f"Label: {obj[new_objs[-1]]} " + reminder_text
        text_size, _ = cv2.getTextSize(reminder_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        DrawText(expand_image, reminder_text, font_thickness=font_thickness, font_scale=font_scale, pos='tl', axis=(0, 0))
        
        if len(new_coords) > 0:
            for i in range(len(new_coords)):
                cv2.rectangle(expand_image, (new_coords[i][0], new_coords[i][1]), 
                              (new_coords[i][2], new_coords[i][3]), (clr[new_objs[i]]), 2)
                DrawText(expand_image, obj[new_objs[i]], font_thickness=2, font_scale=1, pos='tl', axis= (new_coords[i][0], new_coords[i][1]))
        
        cv2.imshow(param[4], expand_image)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        expand_image = handle_right_buttom_up_normal(param)
        cv2.imshow(param[4], expand_image)


   
def add_boarder(img, border_height=45):
    new_height = img.shape[0] + border_height
    expand_image = np.zeros((new_height, img.shape[1], 3), dtype=np.uint8)
    expand_image[:img.shape[0], :, :] = img
    return expand_image
   


def initialize_parameters(last_time_num, source):
    source = str(source)
    if last_time_num is None:
        img_count = 0
    else:
        img_count = max(0, last_time_num)
    return source, img_count


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device}')
    return device


def load_model(weights, device):
    model = attempt_load(weights, device=device)
    model.eval()
    return model


def get_image_files(source_dir):
    files = sorted(os.listdir(Path(source_dir))) # 得到圖片列表
    img_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(img_files)
    return img_files, total_images


def get_image_path_and_name(source, img_files, img_count, save_dir):
    img_path = os.path.join(source, img_files[img_count]) # 圖片路徑
    file_name, _ = os.path.splitext(img_files[img_count])
    sav_img = os.path.join(save_dir, file_name + '.txt')
    return img_path, file_name, sav_img


def read_and_preprocess_image(img_path, imagesz, device):
    im0 = cv2.imread(img_path)
    im_tmp = copy.deepcopy(im0)
    im = letterbox(im0, imagesz)[0]
    im = im.transpose((2, 0, 1))[::-1].copy()
    im = torch.from_numpy(im).float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    im = im.to(device)
    return im, im_tmp


def predict_image(im, model, conf_thres, iou_thres, max_det, im0_shape):
    pred = model(im)
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
    pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0_shape).round()
    return pred



def draw_prediction_on_image(pred, im0, obj, clr):
    expand_image = add_boarder(im0)
    font_thickness=2
    font_scale=0.58
    lst = []
    
    for i in range(len(pred[0])):
        x = int(pred[0][i][0])
        y = int(pred[0][i][1])
        w = int(pred[0][i][2]) - int(pred[0][i][0])
        h = int(pred[0][i][3]) - int(pred[0][i][1])
        classes = int(pred[0][i][5])
        if classes != 0:
            continue
        cv2.rectangle(expand_image, (int(pred[0][i][0]), int(pred[0][i][1])), (int(pred[0][i][2]), int(pred[0][i][3])), clr[classes], 3, cv2.LINE_AA)
        DrawText(expand_image, obj[classes], font_thickness=2, font_scale=1, pos='tl', axis=(int(pred[0][i][0]), int(pred[0][i][1])))
        lst_i = location(x, y, w, h, im0.shape[1], im0.shape[0], classes)
        lst.append(lst_i)
        
    word = "[Space]: Edit | [Enter]: Save | [Esc]: Exit | [A]: Prev | [D]: Skip"
    text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_w, text_h = text_size
    DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, im0.shape[0] + text_h * 2))
    return expand_image, lst

def process_user_input(key, save_img, lst, im_tmp, file_name, flag=1):
    action = get_key_action(key)
    if action == 'save': 
        with open(save_img, 'w+') as f:
            for i in range(len(lst)):
                for j in range(len(lst[i])):
                    if j == len(lst[i])-1:
                        f.write(str(lst[i][j]))
                        f.write('\n')
                    else:
                        f.write(str(lst[i][j]))
                        f.write(' ')
        print('Label Saved Successfully')
        cv2.destroyAllWindows()
        return True, False, None
    elif action == 'modify':
        cv2.destroyAllWindows()
        lst_obj = []
        lst_a = [[], im_tmp, lst_obj, 0, file_name]
        im_new = copy.deepcopy(im_tmp)
        expand_image = add_boarder(im_new)
        font_thickness=2
        font_scale=0.7
        if flag == 1:
            word = "[Enter]: Save | [Esc]: Exit | [A]: Prev | [D]: Skip"
        elif flag == 2:
            word = f"Label: {obj[0]} | [Enter] Save | [Esc] Exit | [A] Prev | [D] Skip"
        text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, im_tmp.shape[0] + text_h * 2))
        cv2.imshow(file_name, expand_image)
        if flag == 1:
            cv2.setMouseCallback(file_name, show_xy, lst_a)
        elif flag == 2:
            cv2.setMouseCallback(file_name, show_xy_normal_multilabel, lst_a)
        return False, True, lst_a
    return False, False, None


def get_plate_string():
    while True:
        plate_string = input('Please enter the license plate (3 letters + 4 numbers): ').upper()

        if re.match(r'^[A-Z]{3}\d{4}$', plate_string):
            return [obj.index(char) for char in plate_string]
        else:
            print('Incorrect format. Please re-enter.')
      
            
def process_image_annotations(lst_a, im0):
    lst = []
    for i in range(len(lst_a[0])):
        x0, x1 = sorted([int(lst_a[0][i][0]), int(lst_a[0][i][2])])
        y0, y1 = sorted([int(lst_a[0][i][1]), int(lst_a[0][i][3])])
        w, h = x1 - x0, y1 - y0
        classes = lst_a[2][i]
        lst_i = location(x0, y0, w, h, im0.shape[1], im0.shape[0], classes)
        lst.append(lst_i)
    return lst


def save_labels_to_file(lst, save_img):
    with open(save_img, 'w+') as f:
        for entry in lst:
            f.write(" ".join(map(str, entry)) + "\n")
    print('Label Saved Successfully')
    
        

def get_key_action(key):
    return key_actions.get(key, 'invalid key')



def update_label_and_display(action, count, obj, lst_a, im_tmp, file_name):
    try:
        if action =='switch_next':
            count += 1
        elif action == 'switch_prev':
            count -= 1
            
        item = count % len(obj)
        lst_a[3] = item
        im_new = copy.deepcopy(im_tmp)
        expand_image = add_boarder(im_new)
        font_thickness=2
        font_scale=0.5
        word = f"Label: {obj[item]} | [S]: Next Obj | [W] Prev obj | [Enter]: Save | [B]: Undo | [ESC]: Quit"
        text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, im_new.shape[0] + text_h * 2))

        for i in range(len(lst_a[0])):
            cv2.rectangle(expand_image, (lst_a[0][i][0], lst_a[0][i][1]), (lst_a[0][i][2], lst_a[0][i][3]), (clr[lst_a[2][i]]), 2)
            DrawText(expand_image, obj[lst_a[2][i]], font_thickness=2, font_scale=1, pos='tl', axis=(lst_a[0][i][0], lst_a[0][i][1]))

        cv2.imshow(file_name, expand_image)
    except IndexError:
        print('Index Error')
        pass
    return count


  
   
def run_normal_multilabel(
    last_time_num = None,
    weights = './runs/train/exp3/weights/best.pt',
    source = './data/images/', 
    imagesz = (640, 640),
    conf_thres = 0.25,
    iou_thres = 0.45,
    max_det = 1000,
    store = './data/labels_test/',
):
    source, img_count = initialize_parameters(last_time_num, source)
    device = get_device()
    model = load_model(weights, device)
    img_files, total_images = get_image_files(source)
    
    while img_count < total_images:
        img_path, file_name, save_img = get_image_path_and_name(source, img_files, img_count, store)
        print('Current Image Path: ', img_path)
        im, im_tmp = read_and_preprocess_image(img_path, imagesz, device)
        pred = predict_image(im, model, conf_thres, iou_thres, max_det, im_tmp.shape)
        
        expand_image, lst = draw_prediction_on_image(pred, im_tmp, obj, clr)
        cv2.imshow(file_name, expand_image)
        key = cv2.waitKey(0)
        action = get_key_action(key)
        while True:
            is_next_image, is_reannotate, lst_a = process_user_input(key, save_img, lst, im_tmp, file_name, flag=2)
            if is_next_image:
                img_count += 1
                cv2.destroyAllWindows()
                break
            elif is_reannotate:
                count = 0
                while True:
                    key = cv2.waitKey(0)
                    action = get_key_action(key)
                    if action == 'save': 
                        lst = process_image_annotations(lst_a, im_tmp)
                        save_labels_to_file(lst, save_img)
                        img_count += 1
                        cv2.destroyAllWindows()
                        break

                    elif action in ['switch_next', 'switch_prev']:
                        count = update_label_and_display(action, count, obj, lst_a, im_tmp, file_name) 

                          
                    
                    elif action == 'pass':  
                        img_count += 1
                        cv2.destroyAllWindows()
                        break
                    elif action == 'exit':  
                        cv2.destroyAllWindows()
                        print('Program Terminated')
                        quit()
                    elif action == 'previous':  
                        img_count -= 1
                        img_count = max(0, img_count)
                        cv2.destroyAllWindows()
                        break
                    else:
                        print('Input error. Please re-enter.')
                        continue
                break
                
                    
            
            elif action == 'pass': 
                img_count += 1
                cv2.destroyAllWindows()
                break
            elif action == 'exit': 
                cv2.destroyAllWindows()
                print('Program Terminated')
                quit()
            elif action == 'previous':
                img_count -= 1
                if img_count < 0:
                    img_count = 0
                cv2.destroyAllWindows()
                break
            else:
                print('Input error. Please re-enter.')
                break
            
        cv2.destroyAllWindows()   

   
                    

def run(
    last_time_num = None,
    weights = './runs/train/exp3/weights/best.pt',
    source = './data/images/', 
    imagesz = (640, 640),
    conf_thres = 0.25,
    iou_thres = 0.45,
    max_det = 1000,
    store = './data/labels_test/',
):
    source, img_count = initialize_parameters(last_time_num, source)
    device = get_device()
    model = load_model(weights, device)
    img_files, total_images = get_image_files(source)
    
    while img_count < total_images:
        img_path, file_name, save_img = get_image_path_and_name(source, img_files, img_count, store)
        print('Current Image Path: ', img_path)
        im, im_tmp = read_and_preprocess_image(img_path, imagesz, device)
        pred = predict_image(im, model, conf_thres, iou_thres, max_det, im_tmp.shape)
        
        expand_image, lst = draw_prediction_on_image(pred, im_tmp, obj, clr)
        cv2.imshow(file_name, expand_image)
        key = cv2.waitKey(0)
        action = get_key_action(key)
        while True:
            is_next_image, is_reannotate, lst_a = process_user_input(key, save_img, lst, im_tmp, file_name, flag=1)
            if is_next_image:
                img_count += 1
                cv2.destroyAllWindows()
                break
            elif is_reannotate:
                while True:
                    key = cv2.waitKey(0)
                    action = get_key_action(key)
                    if action == 'save': 
                        if len(lst_a[0]) != 7:
                            print("Label count hasn't reached the maximum (Please label 7).")
                        else:
                            lst_a[2] = get_plate_string()
                            lst_a[0] = sorted(lst_a[0], key=lambda x: x[0])
                            lst = process_image_annotations(lst_a, im_tmp)
                            save_labels_to_file(lst, save_img)
                            img_count += 1
                            cv2.destroyAllWindows()
                            break
                    elif action == 'pass':  
                        img_count += 1
                        cv2.destroyAllWindows()
                        break
                    elif action == 'exit':  
                        cv2.destroyAllWindows()
                        print('Program Terminated')
                        quit()
                    elif action == 'previous':  
                        img_count -= 1
                        img_count = max(0, img_count)
                        cv2.destroyAllWindows()
                        break
                    else:
                        print('Input error. Please re-enter.')
                        continue
                break
                
                    
            
            elif action == 'pass': 
                img_count += 1
                cv2.destroyAllWindows()
                break
            elif action == 'exit': 
                cv2.destroyAllWindows()
                print('Program Terminated')
                quit()
            elif action == 'previous': 
                img_count -= 1
                if img_count < 0:
                    img_count = 0
                cv2.destroyAllWindows()
                break
            else:
                print('Input error. Please re-enter.')
                break
            
        cv2.destroyAllWindows()



def run_label_no_model_ANPR(
    last_time_num = None,
    source = './data/images/', 
    store = './data/labels_test/',
):
    source, img_count = initialize_parameters(last_time_num, source)
    img_files, total_images = get_image_files(source)
    
    while img_count < total_images:
        img_path, file_name, save_img = get_image_path_and_name(source, img_files, img_count, store)
        print('Current Image Path: ', img_path)
        im0 = cv2.imread(img_path)
        im_tmp = copy.deepcopy(im0)
        expand_image = add_boarder(im_tmp)
        key = cv2.waitKey(0)
        action = get_key_action(key)
        
        lst_obj = []
        lst_a = [[], im_tmp, lst_obj, 0, file_name]
        im_new = copy.deepcopy(im_tmp)
        expand_image = add_boarder(im_new)
        font_thickness=2
        font_scale=0.6
        word = "[Enter]: Save | [R]: Undo | [A]: Prev | [D]: Skip"
        text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, im_tmp.shape[0] + text_h * 2))
        cv2.imshow(file_name, expand_image)
        cv2.setMouseCallback(file_name, show_xy, lst_a)
        while True:
            key = cv2.waitKey(0)
            action = get_key_action(key)
            if action == 'save': 
                if len(lst_a[0]) != 7:
                    print("Label count hasn't reached the maximum (Please label 7).")
                else:
                    lst_a[2] = get_plate_string()
                    lst_a[0] = sorted(lst_a[0], key=lambda x: x[0])
                    lst = process_image_annotations(lst_a, im_tmp)
                    save_labels_to_file(lst, save_img)
                    img_count += 1
                    cv2.destroyAllWindows()
                    break
            elif action == 'pass':  
                img_count += 1
                cv2.destroyAllWindows()
                break
            elif action == 'exit':  
                cv2.destroyAllWindows()
                print('Program Terminated')
                quit()
            elif action == 'previous':  
                img_count -= 1
                img_count = max(0, img_count)
                cv2.destroyAllWindows()
                break
            else:
                print('Input error. Please re-enter.')
                continue
            break


def run_label_no_model_normal_multilabel(
    last_time_num = None,
    source = './data/images/', 
    store = './data/labels_test/',
):
    source, img_count = initialize_parameters(last_time_num, source)
    img_files, total_images = get_image_files(source)
    
    while img_count < total_images:
        img_path, file_name, save_img = get_image_path_and_name(source, img_files, img_count, store)
        print('Current Image Path: ', img_path)
        im0 = cv2.imread(img_path)
        im_tmp = copy.deepcopy(im0)
        expand_image = add_boarder(im_tmp)
        key = cv2.waitKey(0)
        action = get_key_action(key)
        
        lst_obj = []
        lst_a = [[], im_tmp, lst_obj, 0, file_name]
        im_new = copy.deepcopy(im_tmp)
        expand_image = add_boarder(im_new)
        font_thickness=2
        font_scale=0.5
        word = "[Enter]: Save | [R]: Undo | [A]: Prev | [D]: Skip | [S]: Next obj | [W]: Prev obj"
        text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        DrawText(expand_image, word, font_thickness=font_thickness, font_scale=font_scale, pos='bl', axis=(0, im_tmp.shape[0] + text_h * 2))
        cv2.imshow(file_name, expand_image)
        cv2.setMouseCallback(file_name, show_xy_normal_multilabel, lst_a)
        count = 0
        while True:
            key = cv2.waitKey(0)
            action = get_key_action(key)
            if action == 'save': 
                lst = process_image_annotations(lst_a, im_tmp)
                save_labels_to_file(lst, save_img)
                img_count += 1
                cv2.destroyAllWindows()
                break
                
            elif action in ['switch_next', 'switch_prev']:
                count = update_label_and_display(action, count, obj, lst_a, im_tmp, file_name)
            
            
            elif action == 'pass':  
                img_count += 1
                cv2.destroyAllWindows()
                break
            elif action == 'exit':  
                cv2.destroyAllWindows()
                print('Program Terminated')
                quit()
            elif action == 'previous':  
                img_count -= 1
                img_count = max(0, img_count)
                cv2.destroyAllWindows()
                break
            else:
                print('Input error. Please re-enter.')
                continue
        

    
if __name__ == '__main__':
    config_manager = ConfigManager()
    conf = config_manager.get_config()
    
    global drawing
    drawing = conf['drawing']
    
    global obj
    obj = conf['obj']
    
    global clr
    clr = conf['clr']
    
    key_actions = conf['key_actions']
    
    
    parser = argparse.ArgumentParser(description='Your script description')
    
    parser.add_argument('--mode', type=str, choices=['normal', 'LPR'], required=True, help='Choose a mode to run the script(normal_multilabel, multilabel_for_ANPR, label_with_no_model_ANPR, label_with_no_model_normal_multilabel)')
    parser.add_argument('--last_time_num', type=int, default=None, help='Last time you stop at which image')
    parser.add_argument('--weights', type=str, default=None, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--imagesz', type=int, nargs=2, default=(640, 640), help='image size')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--store', type=str, default='./labels_test/', help='store')
    
    args = parser.parse_args()
    
    # 判斷source是否為資料夾
    if not os.path.isdir(args.source):
        print(f"Error: '{args.source}' is not a valid directory. Please provide a valid directory path.")
        exit(1)
        
    # 判斷weights是否為檔案
    if args.weights and not os.path.isfile(args.weights):
        print(f"Error: '{args.weights}'  is not a valid file. Please provide a valid file path.")
        exit(1)
        
    # 判斷store是否為資料夾
    if not os.path.isdir(args.store):
        print(f"Error: '{args.store}' is not a valid directory. Please provide a valid directory path.")
        exit(1)
    

    
    if args.mode == 'normal':
        if args.weights:
            run_normal_multilabel(
                last_time_num=args.last_time_num,
                weights=args.weights,
                source=args.source,
                imagesz=tuple(args.imagesz),
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                max_det=args.max_det,
                store=args.store
            )
        else:
           
            run_label_no_model_normal_multilabel(
                last_time_num=args.last_time_num,
                source=args.source,
                store=args.store
            )
    elif args.mode == 'LPR':
        if args.weights:
            run(
                last_time_num=args.last_time_num,
                weights=args.weights,
                source=args.source,
                imagesz=tuple(args.imagesz),
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                max_det=args.max_det,
                store=args.store
            )
        else:
            run_label_no_model_ANPR(
                last_time_num=args.last_time_num,
                source=args.source,
                store=args.store
            )

    