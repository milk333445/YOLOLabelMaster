# YOLOLabelMaster
YOLOLabelMaster is a powerful tool designed specifically for rapid annotation of YoloV5 datasets. It not only supports the labeling of general multi-class objects and the drawing of bounding boxes but also comes with advanced automatic annotation features, enabling users to annotate the required objects more swiftly. This tool goes beyond typical object labeling as it includes specialized support for annotating license plate characters, making it easy to mark characters on license plates.

## Key Features:
1. Swift Multi-Class Labeling: YOLOLabelMaster offers an intuitive and efficient interface, allowing you to easily label multiple object classes and draw bounding boxes to create your YoloV5 dataset.
2. Advanced Automatic Annotation: With the assistance of the YoloV5 model you upload, YOLOLabelMaster possesses robust automatic annotation capabilities. It can automatically detect objects in images and generate preliminary labels, saving you valuable time.
3. License Plate Character Annotation: In addition to general object labeling, this tool provides convenient functionality for annotating characters on license plates, ensuring more precise recognition.
4. Editing and Saving: YOLOLabelMaster enables you to effortlessly edit automatically generated labels to ensure accuracy. Once your labeling is complete, you can conveniently save your work.

Whether you are a professional object recognition engineer or a beginner, YOLOLabelMaster offers a high-efficiency, user-friendly, and feature-rich tool to expedite the process of creating YoloV5 datasets. No longer do you need to spend extensive time manually annotating; now, simply use YOLOLabelMaster as your ultimate labeling assistant.

## How to Use
1. First, you need to git clone https://github.com/milk333445/YOLOLabelMaster.git
2. Download the necessary Python packages
```python=
git clone https://github.com/milk333445/YOLOLabelMaster.git
cd YOLOLabelMaster
pip install -r requirements.txt
```

## Quick Start: Prepare Your Data
1. Before you begin, please make sure you have prepared a folder containing the image dataset you want to label (it must contain .jpg, .jpeg, or .png image files). Note that any other file formats will be automatically ignored by the program.
2. Prepare an empty folder that will be used to store your label data files (.txt). The program will generate corresponding labels in this folder.
3. If you plan to use the YoloV5 model for automatic labeling, ensure that you have prepared the relevant YoloV5 weight files. You will need to pass them as parameters in the subsequent steps.

## Quick Start: Configure autolabel_settings.yaml
1. Before using this tool, you need to configure autolabel_settings.yaml to set the object classes you want to label. You can define class names by editing the "classes" field in autolabel_settings.yaml. Please be aware that class names will be encoded in the order they appear in the file, such as 0, 1, 2, and so on, so pay special attention to this part.
```python=
classes:
  - A # 0
  - B # 1
  - C # 2
  - D # 3
  、
  、
  、
```

2. Additionally, you can customize "key_actions" based on your personal preferences to configure custom keyboard triggers for adjusting labeling methods to meet your needs.
```python=
key_actions:
  13: 'save' # enter
  32: 'modify' # space
  27: 'exit' # esc
  100: 'pass' # d
  68: 'pass' # D
  65: 'previous' # a
  97: 'previous' # A
  119: 'switch_prev' # w
  115: 'switch_next' # s
  87: 'switch_prev' # W
  83: 'switch_next' # S
```

## Quick Start: Launching the Main Program and Setting Parameters
```python=
python main.py --mode [choose mode] --last_time_num [last image number] --weights [model weights file path] --source [image folder path] --imagesz [image width] [image height] --conf_thres [object confidence threshold] --iou_thres [NMS IOU threshold] --max_det [maximum detections per image] --store [label storage path]
```
- You don't need to configure all parameters every time. Here are detailed explanations of some important parameters such as mode, last_time_num, weights, source, and store:
  - mode (Mode):
    - normal (Normal Mode): This is the general mode for multi-class labeling where you can quickly switch object categories using the keyboard.
    - LPR (License Plate Mode): This mode is designed for annotating license plate characters (default is 3 English characters + 4 numeric characters). In this mode, you will need to input all labels via the terminal.
  - last_time_num (Last Image Number):
    - This parameter allows you to quickly jump to a specific image number. If you have previously labeled the first five images, you can input last_time_num 6 to directly jump to labeling the sixth image.
  - weights (Model Weights File):
    - If you wish to perform pre-labeling using the YoloV5 model, provide the path to the weight file. The program will automatically switch to pre-labeling mode and give you the option to modify labels or save them directly. 
  - source (Image Folder Path):
    - This is a mandatory parameter that specifies the path to the folder containing the images you want to label. 
  - store (Label Storage Path):
    - This is also a mandatory parameter that specifies where you want to save the label information.
   
Using these parameters, you can easily configure the main program to label your image dataset and perform labeling in different modes according to your needs.









