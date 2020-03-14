import os, sys
import skimage
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

ROOT_DIR = os.path.abspath(os.getcwd())
LOG_DIR = os.path.join(ROOT_DIR, "logs")
COCO_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
IMAGE_PATH = 'catdog2.jpg'

class InferenceConfig(Config):
    NAME = 'COCO'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    NUM_CLASSES = 1 + 80

config = InferenceConfig()

modelo = modellib.MaskRCNN(mode="inference", config=config, model_dir=LOG_DIR)

caminho_pesos = COCO_PATH
if not os.path.exists(caminho_pesos):
     utils.download_trained_weights(caminho_pesos)
    
modelo.load_weights(caminho_pesos, by_name=True)

image = skimage.io.imread(IMAGE_PATH)
r = modelo.detect([image], verbose = 1)[0]

for classe in r['class_ids']:
    if((classe != 16) and (classe != 17)):
        i = np.where(r['class_ids'] == classe)[0]
        r['class_ids'] = np.delete(r['class_ids'], i, 0)
        r['masks'] = np.delete(r['masks'], i, 2)
        r['rois'] = np.delete(r['rois'], i, 0)
        r['scores'] = np.delete(r['scores'], i, 0)

for i in range(len(r['class_ids'])):
    r['class_ids'][i] -= 15

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            ['BG', 'Gato', 'Cachorro'], r['scores'])
