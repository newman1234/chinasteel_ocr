import os
from shutil import copy2

import pandas as pd
import numpy as np
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

path_img_folder = 'data/public_training_data/public_training_data/'
path_img_folder_test = 'data/public_training_data/public_testing_data/'
path_img_folder_public_test = 'data/public_testing_data/'
path_img_folder_private_test = '../private_data_v2/'

path_label = 'data/標記與資料說明/Training Label/public_training_data.csv'
path_new_label = 'data/標記與資料說明/public_data_label 更新表.xlsx'

batch_size = 256

cfg = get_cfg()
# COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("img_train",)
# cfg.DATASETS.TEST = ("img_val",)
cfg.DATALOADER.NUM_WORKERS = 2
# COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR 0.00025 
cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[128, 256, 512]]  # [[32, 64, 128, 256, 512]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5]] # [[0.5, 1.0, 2.0]]

# copy2(f'{path_prefix}output/model.pth', os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

# cfg.INPUT.RANDOM_FLIP = "none"  # default: horizontal

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

def inference_img(path_input, path_output, df=None):

    list_img = []
    for f in tqdm(os.listdir(path_input)):
        im = cv2.imread(path_input + f)
        # print(im.shape)
        outputs = predictor(im)
        pred_boxes = outputs["instances"].to("cpu").pred_boxes.tensor
        pred_boxes = np.array(pred_boxes)
        n = pred_boxes.shape[0]
        label = df.loc[f, 'label']

        if n == 1:
            x1 = int(pred_boxes[0, 1])
            x2 = int(pred_boxes[0, 3])+1
            y1 = int(pred_boxes[0, 0])
            y2 = int(pred_boxes[0, 2])+1
            # print(y1, y2, x1, x2)
            roi = im[x1:x2, y1:y2, ::-1]
            cv2.imwrite(os.path.join(path_output, f'{f[:f.index(".")]}_{label}.png'), roi)

        elif n >= 2:
            ### postprocess to get better detection box
            x1, y1, x2, y2 = int(np.min(pred_boxes[:, 0])), int(np.min(pred_boxes[:, 1])), int(np.max(pred_boxes[:, 2]))+1, int(np.max(pred_boxes[:, 3]))+1

            ### postprocess to get better detection box
            roi = im[y1:y2, x1:x2, ::-1]
            cv2.imwrite(os.path.join(path_output, f'{f[:f.index(".")]}_{label}.png'), roi)
            print(os.path.join(path_output, f'{f[:f.index(".")]}_{label}.png'))
        
    return 

# df_test = pd.read_csv('data/標記與資料說明/Training Label/public_testing_data.csv')
# df_test['filename'] = df_test['filename'] + '.jpg'
# df_test = df_test.set_index('filename')
# df_test

# # output test crop images
# for f in tqdm(os.listdir(path_img_folder_test)):
#     im = cv2.imread(path_img_folder_test + f)
#     # print(im.shape)
#     outputs = predictor(im)
#     pred_boxes = outputs["instances"].to("cpu").pred_boxes.tensor
#     pred_boxes = np.array(pred_boxes)
#     n = pred_boxes.shape[0]
#     label = df_test.loc[f, 'label']

#     if n == 1:
#         # v = Visualizer(im[:, :, ::-1],
#         #     metadata=digit_metadata, 
#         #     scale=0.5 
#         # )
#         # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         # cv2_imshow(out.get_image()[:, :, ::-1])

#         # print(pred_boxes)
#         x1 = int(pred_boxes[0, 1])
#         x2 = int(pred_boxes[0, 3])+1
#         y1 = int(pred_boxes[0, 0])
#         y2 = int(pred_boxes[0, 2])+1
#         # print(y1, y2, x1, x2)
#         roi = im[x1:x2, y1:y2, ::-1]
#         cv2.imwrite(f'output/for_recog/test/{f[:f.index(".")]}_{label}.png', roi)
#         # cv2_imshow(roi)

#     elif n >= 2:
#         # print(f)
#         # v = Visualizer(im[:, :, ::-1],
#         #     metadata=digit_metadata, 
#         #     scale=0.5 
#         # )
#         # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         # cv2_imshow(out.get_image()[:, :, ::-1])

#         ### postprocess to get better detection box
#         x1, y1, x2, y2 = int(np.min(pred_boxes[:, 0])), int(np.min(pred_boxes[:, 1])), int(np.max(pred_boxes[:, 2]))+1, int(np.max(pred_boxes[:, 3]))+1

#         ### postprocess to get better detection box
#         roi = im[y1:y2, x1:x2, ::-1]
#         cv2.imwrite(f'output/for_recog/test/{f[:f.index(".")]}_{label}.png', roi)
#         print(f'output/for_recog/test/{f[:f.index(".")]}_{label}.png')
#         # input()

# df_train = pd.read_csv('data/標記與資料說明/Training Label/public_training_data.csv')
# df_train = df_train.set_index('filename')
# df_train_new = pd.read_excel('data/標記與資料說明/public_data_label 更新表.xlsx', index_col=0)[['更新的label']].rename(columns={'更新的label':'label'})
# df_train.loc[df_train_new.index, 'label'] = df_train_new['label']
# df_train = df_train.reset_index()
# df_train['filename'] = df_train['filename'] + '.jpg'
# df_train = df_train.set_index('filename')
# df_train

# # output train crop images
# for f in tqdm(os.listdir(path_img_folder)):
#     im = cv2.imread(path_img_folder + f)
#     # print(im.shape)
#     outputs = predictor(im)
#     pred_boxes = outputs["instances"].to("cpu").pred_boxes.tensor
#     pred_boxes = np.array(pred_boxes)
#     n = pred_boxes.shape[0]
#     label = df_train.loc[f, 'label']

#     if n == 1:
#         # v = Visualizer(im[:, :, ::-1],
#         #     metadata=digit_metadata, 
#         #     scale=0.5 
#         # )
#         # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         # cv2_imshow(out.get_image()[:, :, ::-1])

#         # print(pred_boxes)
#         x1 = int(pred_boxes[0, 1])
#         x2 = int(pred_boxes[0, 3])+1
#         y1 = int(pred_boxes[0, 0])
#         y2 = int(pred_boxes[0, 2])+1
#         # print(y1, y2, x1, x2)
#         roi = im[x1:x2, y1:y2, ::-1]
#         cv2.imwrite(f'output/for_recog/train/{f[:f.index(".")]}_{label}.png', roi)
#         # cv2_imshow(roi)

#     elif n >= 2:
#         # print(f)
#         # v = Visualizer(im[:, :, ::-1],
#         #     metadata=digit_metadata, 
#         #     scale=0.5 
#         # )
#         # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         # cv2_imshow(out.get_image()[:, :, ::-1])

#         ### postprocess to get better detection box
#         x1, y1, x2, y2 = int(np.min(pred_boxes[:, 0])), int(np.min(pred_boxes[:, 1])), int(np.max(pred_boxes[:, 2]))+1, int(np.max(pred_boxes[:, 3]))+1

#         ### postprocess to get better detection box
#         roi = im[y1:y2, x1:x2, ::-1]
#         cv2.imwrite(f'output/for_recog/train/{f[:f.index(".")]}_{label}.png', roi)
#         # input()
#         print(f'output/for_recog/train/{f[:f.index(".")]}_{label}.png')

# output public test crop images
# for f in tqdm(os.listdir(path_img_folder_public_test)):
#     im = cv2.imread(path_img_folder_public_test + f)
#     # print(im.shape)
#     outputs = predictor(im)
#     pred_boxes = outputs["instances"].to("cpu").pred_boxes.tensor
#     pred_boxes = np.array(pred_boxes)
#     n = pred_boxes.shape[0]

#     if n == 1:
#         # v = Visualizer(im[:, :, ::-1],
#         #     metadata=digit_metadata, 
#         #     scale=0.5 
#         # )
#         # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         # cv2_imshow(out.get_image()[:, :, ::-1])

#         # print(pred_boxes)
#         x1 = int(pred_boxes[0, 1])
#         x2 = int(pred_boxes[0, 3])+1
#         y1 = int(pred_boxes[0, 0])
#         y2 = int(pred_boxes[0, 2])+1
#         # print(y1, y2, x1, x2)
#         roi = im[x1:x2, y1:y2, ::-1]
#         cv2.imwrite(f'output/for_recog/public_test/{f[:f.index(".")]}.png', roi)
#         # cv2_imshow(roi)

#     elif n >= 2:
#         # print(f)
#         # v = Visualizer(im[:, :, ::-1],
#         #     metadata=digit_metadata, 
#         #     scale=0.5 
#         # )
#         # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         # cv2_imshow(out.get_image()[:, :, ::-1])

#         ### postprocess to get better detection box
#         x1, y1, x2, y2 = int(np.min(pred_boxes[:, 0])), int(np.min(pred_boxes[:, 1])), int(np.max(pred_boxes[:, 2]))+1, int(np.max(pred_boxes[:, 3]))+1

#         ### postprocess to get better detection box
#         roi = im[y1:y2, x1:x2, ::-1]
#         cv2.imwrite(f'output/for_recog/public_test/{f[:f.index(".")]}.png', roi)
#         print(f'output/for_recog/public_test/{f[:f.index(".")]}.png')
#         # input()




for f in tqdm(os.listdir(path_img_folder_private_test)):
    im = cv2.imread(path_img_folder_private_test + f)
    # print(im.shape)
    outputs = predictor(im)
    pred_boxes = outputs["instances"].to("cpu").pred_boxes.tensor
    pred_boxes = np.array(pred_boxes)
    n = pred_boxes.shape[0]

    if n == 1:
        # v = Visualizer(im[:, :, ::-1],
        #     metadata=digit_metadata, 
        #     scale=0.5 
        # )
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])

        # print(pred_boxes)
        x1 = int(pred_boxes[0, 1])
        x2 = int(pred_boxes[0, 3])+1
        y1 = int(pred_boxes[0, 0])
        y2 = int(pred_boxes[0, 2])+1
        # print(y1, y2, x1, x2)
        roi = im[x1:x2, y1:y2, ::-1]
        cv2.imwrite(f'output/for_recog/private_test/{f[:f.index(".")]}.png', roi)
        # cv2_imshow(roi)

    elif n >= 2:
        # print(f)
        # v = Visualizer(im[:, :, ::-1],
        #     metadata=digit_metadata, 
        #     scale=0.5 
        # )
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])

        ### postprocess to get better detection box
        x1, y1, x2, y2 = int(np.min(pred_boxes[:, 0])), int(np.min(pred_boxes[:, 1])), int(np.max(pred_boxes[:, 2]))+1, int(np.max(pred_boxes[:, 3]))+1

        ### postprocess to get better detection box
        roi = im[y1:y2, x1:x2, ::-1]
        cv2.imwrite(f'output/for_recog/private_test/{f[:f.index(".")]}.png', roi)
        print(f'output/for_recog/private_test/{f[:f.index(".")]}.png')
        # input()