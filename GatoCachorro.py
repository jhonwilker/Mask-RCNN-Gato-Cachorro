import os
import skimage
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

# Configuração de inferência do modelo, sub-classe da classe Config
class InferenceConfig(Config):
    NAME = 'COCO'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    NUM_CLASSES = 1 + 80

# Detecta os gatos e cachorros e salva em um arquivo .png
def detectar_classes(modelo, arquivo_imagem):
    # Transforma a imagem em uma matriz que possa ser lida pelo modelo    
    imagem = skimage.io.imread(arquivo_imagem)
    
    # Detecta e retorna as Classes, Masks, Regiões de Interesse e 
    # Índices de Confiança
    r = modelo.detect([imagem], verbose = 0)[0]
    
    # O modelo Coco originalmente detecta um total de 80 classes, incluindo
    # gatos e cachorros, e nessa parte do código eu removo todasas detecções
    # não relacionadas a eles
    for classe in r['class_ids']:
        if((classe != 16) and (classe != 17)):
            i = np.where(r['class_ids'] == classe)[0]
            r['class_ids'] = np.delete(r['class_ids'], i, 0)
            r['masks'] = np.delete(r['masks'], i, 2)
            r['rois'] = np.delete(r['rois'], i, 0)
            r['scores'] = np.delete(r['scores'], i, 0)
    
    for i in range(len(r['class_ids'])):
        r['class_ids'][i] -= 15
    
    # Salva o arquivo com todos os gatos e cachorros detectados com o mesmo 
    # nome + _detectado no final
    # O método display_instances originalmente apenas mostra um plot do 
    # matplotlib, mas eu modifiquei a mesma para tambem salvar em uma imagem
    nome_arquivo = '.'.join(arquivo_imagem.split('.')[:-1])
    visualize.display_instances(imagem, r['rois'], r['masks'], r['class_ids'], 
                                ['BG', 'Gato', 'Cachorro'], r['scores'],
                                nome_arquivo = nome_arquivo)

# Variáveis padrão para uso
caminho_base = os.path.abspath(os.getcwd())
caminho_Logs = os.path.join(caminho_base, "logs")
caminho_pesos = os.path.join(caminho_base, 'mask_rcnn_coco.h5')

# Carregamento do modelo e pesos
# Caso o arquivo de pesos do dataset Coco não exista na pasta, o mesmo é 
# baixado
modelo = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir=caminho_Logs)
if not os.path.exists(caminho_pesos):
     utils.download_trained_weights(caminho_pesos)
modelo.load_weights(caminho_pesos, by_name=True)

# Se o arquivo for chamado diretamente, requer o argumento --imagem com o nome
# do arquivo
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagem', required=True,
                        metavar="caminho ou URL para uma imagem",
                        help='Imagem para testar e aplicar a detecção')
    
    args = parser.parse_args()
    
    print('Detectando em ' + args.imagem + '...')
    detectar_classes(modelo, args.imagem)
    print("Arquivo salvo em " + '.'.join(args.imagem.split('.')[:-1]) + "_detectado.png")