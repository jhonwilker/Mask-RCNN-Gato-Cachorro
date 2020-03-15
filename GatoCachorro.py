import os
import skimage
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

caminho_base = os.path.abspath(os.getcwd())
caminho_Logs = os.path.join(caminho_base, "logs")
caminho_pesos_coco = os.path.join(caminho_base, 'mask_rcnn_coco.h5')
arquivo_imagem = 'imagem.jpg'

class InferenceConfig(Config):
    NAME = 'COCO'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    NUM_CLASSES = 1 + 80

modelo = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir=caminho_Logs)

caminho_pesos = caminho_pesos_coco
if not os.path.exists(caminho_pesos):
     utils.download_trained_weights(caminho_pesos)
modelo.load_weights(caminho_pesos, by_name=True)

imagem = skimage.io.imread(arquivo_imagem)
r = modelo.detect([imagem], verbose = 0)[0]

for classe in r['class_ids']:
    if((classe != 16) and (classe != 17)):
        i = np.where(r['class_ids'] == classe)[0]
        r['class_ids'] = np.delete(r['class_ids'], i, 0)
        r['masks'] = np.delete(r['masks'], i, 2)
        r['rois'] = np.delete(r['rois'], i, 0)
        r['scores'] = np.delete(r['scores'], i, 0)

for i in range(len(r['class_ids'])):
    r['class_ids'][i] -= 15

nome_arquivo = '.'.join(arquivo_imagem.split('.')[:-1])
visualize.display_instances(imagem, r['rois'], r['masks'], r['class_ids'], 
                            ['BG', 'Gato', 'Cachorro'], r['scores'],
                            nome_arquivo = nome_arquivo)

print("Arquivo salvo em " + nome_arquivo + "_detectado.png")
