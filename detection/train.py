# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import pandas as pd
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

import torch
import random

torch.backends.cudnn.deterministic = True
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

path_img_folder = 'data/public_training_data/public_training_data/'
path_img_folder_test = 'data/public_training_data/public_testing_data/'
path_img_folder_public_test = 'data/public_testing_data/public_testing_data/'

path_label = 'data/標記與資料說明/Training Label/public_training_data.csv'
path_new_label = 'data/標記與資料說明/public_data_label 更新表.xlsx'

# ========== READ LABEL ========== #
df_train = pd.read_csv(path_label)
df_train = df_train.set_index('filename')
df_train_new = pd.read_excel(path_new_label, index_col=0)[['更新的label']].rename(columns={'更新的label':'label'})
df_train.loc[df_train_new.index, 'label'] = df_train_new['label']
df_train['len'] = df_train.label.map(len)
df_train = df_train.reset_index()
print('label 長度分佈\n', df_train.len.value_counts())

df_train, df_val = df_train.iloc[:11000], df_train.iloc[11000:]



# ========== GENERATE INPUT FORMAT ========== #
from detectron2.structures import BoxMode

def euclidean_distance(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def get_img_dicts(img_dir, df):
    dataset_dicts = []
    for row in tqdm(df.itertuples(), total=len(df)):
        record = {}
        
        filename = os.path.join(img_dir, row[1]+'.jpg')
        height, width = cv2.imread(filename, 0).shape
        
        record["file_name"] = filename
        record["image_id"] = row[1]
        record["height"] = height
        record["width"] = width

        if euclidean_distance(row[3], row[4], row[7], row[8]) >= euclidean_distance(row[9], row[10], row[5], row[6]):
            x1, x2 = (int(row[3])+1, int(row[4])), (int(row[7]), int(row[8])+1)
        else:
            x1, x2 = (int(row[9]), int(row[10])), (int(row[5])+1, int(row[6])+1)

        if (x1[0] > x2[0]):
            x1, x2 = (x2[0], x1[1]), (x1[0], x2[1])
      
        objs = [
            {
                "bbox": [x1[0], x1[1], x2[0], x2[1]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0
            }
        ]
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d, df in [("train", df_train), ("val", df_val)]:
    DatasetCatalog.register("img_" + d, lambda df=df: get_img_dicts(path_img_folder, df))
    MetadataCatalog.get("img_" + d).set(thing_classes=["digit"])
digit_metadata = MetadataCatalog.get("img_train")
dataset_dicts = get_img_dicts(path_img_folder, df_train)



# ========== TRAIN MODEL ========== #
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
# COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("img_train",)
cfg.DATASETS.TEST = ("img_val",)
cfg.DATALOADER.NUM_WORKERS = 4
# COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR 0.00025 
cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[128, 256, 512]]  # [[32, 64, 128, 256, 512]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5]] # [[0.5, 1.0, 2.0]]
# cfg.INPUT.RANDOM_FLIP = "none"  # default: horizontal

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
import detectron2.data.transforms as T
# class Trainer(DefaultTrainer):
#     @classmethod
#     def build_train_loader(cls, cfg):
#         mapper=DatasetMapper(cfg, is_train=True, augmentations=[
#             T.RandomBrightness(0.9, 1.1)
#         ])
#         return build_detection_train_loader(cfg, mapper=mapper)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()



# ========== TRAIN MODEL ========== #
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
dataset_dicts_val = get_img_dicts(path_img_folder, df_val)



# ========== CHECK MULTI OUTPUT IMAGES ========== #
cnt_multi_output = 0
for d in tqdm(dataset_dicts_val):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    n = outputs["instances"].to("cpu").pred_boxes.tensor.shape[0]
    # input()
    if n >= 2:
        cnt_multi_output += 1
print(cnt_multi_output, 'images with multi output bbox')


# ========== CHECK MULTI OUTPUT IMAGES ========== #
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("img_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "img_val", num_workers=4)
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`





