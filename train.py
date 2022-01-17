import sys
import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from glob import glob
import warnings
from collections import Counter

from ensemble_boxes import weighted_boxes_fusion
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data.dataloader import default_collate

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator, create_model_from_config
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler
from timm.models import resume_checkpoint, load_checkpoint
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

train = os.listdir('../input/schooldata/school_data/train')
val = os.listdir('../input/schooldata/school_data/val')
test = os.listdir('../input/schooldata/school_data/test')

data = pd.read_csv('../input/schooldata/school_data/final_data.csv')
data.rename(columns={'filename':'image_id', 'x1':'x_min', 'y1':'y_min', 'x2':'x_max', 'y2':'y_max'}, inplace=True)
data.head()

data['split'] = 'na'
data.loc[data[data['image_id'].isin(train)].index, 'split'] = 'train' 
data.loc[data[data['image_id'].isin(val)].index, 'split'] = 'valid'
data.loc[data[data['image_id'].isin(test)].index, 'split']  = 'test'


data['image_path'] = data['image_id'].map(lambda x:os.path.join('../input/school-data2/school_data/schools_annotated',str(x)))
data['class_id'] = [0] * data.shape[0]
data.head()

data = data[~data['image_id'].isin(['42201308.png','61212325.png', '61223303.png'])]
data.reset_index(drop=True, inplace=True)

label2color = [[59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133], [117, 75, 3],
                 [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77], [194, 134, 175],
                 [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100]]

# viz_labels =  ["Aortic_enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
#             "Consolidation", "ILD", "Infiltration", "Lung_Opacity", "Nodule/Mass",
#             "Other_lesion", "Pleural_effusion", "Pleural_thickening", "Pneumothorax",
#             "Pulmonary_fibrosis"]
viz_labels =  ["", "", "", "", ""]


def plot_img(img, size=(18, 18), is_rgb=True, title="", cmap=None):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    #plt.suptitle(title)
    plt.show()

def plot_imgs(imgs, cols=2, size=10, is_rgb=True, title="", cmap=None, img_size=None):
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    return fig
    
def draw_bbox(image, box, label, color):   
    alpha = 0.4
    alpha_font = 0.6
    thickness = 4
    font_size = 2.0
    font_weight = 2
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_weight)[0]
    cv2.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]),
                color, -1)
    cv2.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv2.rectangle(overlay_text, (box[0], box[1]-18-text_height), (box[0]+text_width+8, box[1]),
                (0, 0, 0), -1)
    cv2.addWeighted(overlay_text, alpha_font, output, 1 - alpha_font, 0, output)
    cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                    color, thickness)
    cv2.putText(output, label.upper(), (box[0], box[1]-12),
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_weight, cv2.LINE_AA)
    return output

def draw_bbox_small(image, box, label, color):   
    alpha = 0.4
    alpha_text = 0.4
    thickness = 1
    font_size = 0.4
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
    cv2.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]),
                color, -1)
    cv2.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv2.rectangle(overlay_text, (box[0], box[1]-7-text_height), (box[0]+text_width+2, box[1]),
                (0, 0, 0), -1)
    cv2.addWeighted(overlay_text, alpha_text, output, 1 - alpha_text, 0, output)
    cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                    color, thickness)
    cv2.putText(output, label.upper(), (box[0], box[1]-5),
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


    viz_images = []

for img_id in data['image_id'].unique()[10:20]:
    img_path = data[data.image_id==img_id]['image_path'].iloc[0]
    img_array  = cv2.imread(img_path)

#     img_annotations = df_annotations[df_annotations.image_id==img_id]
#     boxes_actual = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
#     labels_actual = img_annotations['class_id'].to_numpy().tolist()
    
    img_annotations_wbf = data[data.image_id==img_id]
    boxes_wbf = img_annotations_wbf[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    #box_labels_wbf = img_annotations_wbf['class_id'].to_numpy().tolist()
    box_labels_wbf = [0]
    
#     print("Bboxes before WBF:\n", boxes_actual)
#     print("Labels before WBF:\n", labels_actual)
    
#     ## Visualize Original Bboxes
#     img_before = img_array.copy()
#     for box, label in zip(boxes_actual, labels_actual):
#         x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
#         color = label2color[int(label)]
#         img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
#     viz_images.append(img_before)

#     print("Bboxes after WBF:\n", boxes_wbf)
#     print("Labels after WBF:\n", box_labels_wbf)
    
    ## Visualize Bboxes after operation
    img_after = img_array.copy()
    for box, label in zip(boxes_wbf, box_labels_wbf):
        print(box)
        color = label2color[int(label)]
        img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
    viz_images.append(img_after)
        
plot_imgs(viz_images, cmap=None)
#plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
#plt.figtext(0.73, 0.9,"WBF", va="top", ha="center", size=25)
#plt.savefig('wbf.png', bbox_inches='tight')
plt.show()

def get_train_transforms():
    return A.Compose(
        [
        ## RandomSizedCrop not working for some reason. I'll post a thread for this issue soon.
        ## Any help or suggestions are appreciated.
#         A.RandomSizedCrop(min_max_height=(300, 512), height=512, width=512, p=0.5),
#         A.RandomSizedCrop(min_max_height=(300, 1000), height=1000, width=1000, p=0.5),
#         A.OneOf([
#             A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
#                                  val_shift_limit=0.2, p=0.9),
#             A.RandomBrightnessContrast(brightness_limit=0.2, 
#                                        contrast_limit=0.2, p=0.9),
#         ],p=0.9),
#         A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
#         A.OneOf([
#             A.Blur(blur_limit=3, p=1.0),
#             A.MedianBlur(blur_limit=3, p=1.0)
#             ],p=0.1),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.Transpose(p=0.5),
        A.Resize(height=512, width=512, p=1),
#         A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
        ToTensorV2(p=1.0)
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_test_transform():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0)], 
            p=1.0)