from PIL import Image
import os
import numpy as np
from math import sqrt


# Função para converter uma imagem RGBA para RGB (remover o canal alfa)
def convert_rgba_to_rgb(image):
    if image.mode == 'RGBA':
        # Converter para RGB (ignorar o canal alfa)
        image = image.convert('RGB')
    return image

def calcula_rgb_media(image):
    image_np = np.array(image)
    avg_color = image_np.mean(axis=(0,1))
    return avg_color


def found_best_image(target_color, image_colors):
    best_image = None
    min_distance = float('inf')

    for index, img_color in enumerate(image_colors):
        distance = sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(target_color, img_color)))
        if distance < min_distance:
            min_distance = distance
            best_image = index
    
    return best_image

def found_best_image_manhattan(target_color, image_colors):
    best_image = None
    min_distance = float('inf')

    for index, img_color in enumerate(image_colors):
        distance = sum(abs(e1 - e2) for e1, e2 in zip(target_color, img_color))
        if distance < min_distance:
            min_distance = distance
            best_image = index
    
    return best_image

def create_mosaic(dataset, base_image_path, output_image, img_size, fator):
    base_image = Image.open(base_image_path)
    base_width, base_height = base_image.size
    base_image = base_image.resize((base_width*fator, base_height*fator), Image.LANCZOS)
    num_canais = len(base_image.getbands())
    base_width, base_height = base_image.size
    mosaic_shape = (base_height // img_size[0], base_width // img_size[1])
    resized_base_image = base_image.resize((mosaic_shape[1] * img_size[1], mosaic_shape[0] * img_size[0]), Image.LANCZOS)
    
    dataset_images = [os.path.join(dataset, f) for f in os.listdir(dataset) if f.endswith(('png', 'jpg', 'jpeg'))]

    images = [Image.open(img).resize((img_size), Image.LANCZOS) for img in dataset_images]
    images = [convert_rgba_to_rgb(img) for img in images]
    image_colors = [calcula_rgb_media(img) for img in images]
    
    print(f"Num de canais = {num_canais}")
    mosaico = np.zeros((mosaic_shape[0] * img_size[0], mosaic_shape[1] * img_size[1], num_canais), dtype=np.uint8)
    print(f"Imagem final size = {mosaic_shape[0] * img_size[0], mosaic_shape[1] * img_size[1]}")

    used_images = set()
    aux_color, aux_index = [],[]
    for i in range(mosaic_shape[0]):
        for j in range(mosaic_shape[1]):
            bloco = resized_base_image.crop((j*img_size[1], i * img_size[0], (j+1) * img_size[1], (i+1) * img_size[0]))
            target_color = calcula_rgb_media(bloco)

            if len(used_images) > 10:
                for index, color in zip(aux_index, aux_color):
                    image_colors[index] = color
                
                used_images = set()

            best_image = found_best_image(target_color, image_colors)
            while best_image in used_images:
                aux_index.append(best_image)
                aux_color.append(image_colors[best_image])
                image_colors[best_image] = [float('inf')] * 3
                best_image = found_best_image(target_color, image_colors)

            used_images.add(best_image)
            img = images[best_image]
            
            mosaico[i*img_size[0]:(i+1) * img_size[0], j* img_size[1]:(j+1) * img_size[1], :] = img

    mosaico_img = Image.fromarray(mosaico)
    mosaico_img.save(output_image)

"""
    Abre a imagem base e verifica o número de canais na imagem.
    Capturo o tamanho da imagem
    Calculo número de blocos que cabem dentro da imagem atual, baseado no shape do mosaico passado.
    Redimensiono a imagem base para caber o número de blocos calculado.

    Leio todo o dataset de imagens
    Redimensiono cada imagem do dataset para o shape do mosaico passado nos parametros
    Retiro o canal alfa se existir no dataset em cada imagem
    Calculo a média das cores rgb em cada imagem do dataset e armazeno num array de cores.

    Criamos uma imagem zerada com o tamanho calculado anteriormente pro mosaico.


    Lógica de criação do mosaíco:
        Variável used_images, que vai ser responsável por armazenar as imagens que já foram utilizadas no mosaico.
        Um loop que irá percorrer toda a imagem zerada do mosaico afim de inserir as imagens escolhidos no dataset.
        Em cada iteração em (x,y) da imagem, retiramos um bloco de tamanho = mosaic_shape e calculamos a média das cores nesse bloco
        Essa média será usada como objeto de target para encontrarmos uma imagem no dataset.
        Chamamos a funcao que procura a melhor opção de imagem no dataset que corresponde a media de cor da imagem base original.

        A função que busca a melhor imagem, utiliza o algoritmo ou heurística de distância euclidiana, onde o menor valor de distãncia corresponde a melhor opção de imagem.
        Ao encontrar a melhor opção, verificamos se ela já foi utilizada, se sim, retiramos essa imagem o dataset do conjunto de escolhas e executamos a função procuranod por outra imagem.
        Ao encontrar a segunda melhor opção, inserimos essa imagem no bloco de imagem designado e armazemos essa imagem na variável used_images para não ser usada em sequencia.

        Quando o array de used_images bate um limit pre-definido, retornamos todas as imagens do dataset para o conjunto de escolhas, assim, fazendo com que todas as imagens possam ser reutilazadas novamente.
            OBS: Isso evita o reuso excessivo.

        O mosaico é salvo em uma pasta output.


    Discussões:
        1. Quando uma imagem é muito pequena, o shape das imagens do dataset não podem ser muito grandes, o que ocasiona:
            - Imagens pixeladas e pouco nítidas.
            Soluções:
                - Aumentar o size da imagem base original para caber mais blocos de imagens do dataset.
                - Diminuir o shape das imagens do dataset para caber mais blocos de imagens, a imagem fica mais nítida, porém as imagens do dataset mais pixeladas.

        2. Dataset com pouca variedade de cores e formas.
            - Quando temos uma image_base com muitas informações e cores e um dataset fraco, a imagem final falta detalhes. 
"""
    
# Parâmetros do código
base_image = './images_base/fluminense.png'
fator = 4
dataset = './datasets/pokemon'
mosaic_output = 'outputs/mosaico-flu-euclidiano.png'
mosaic_shape = (20,20)

create_mosaic(dataset, base_image, mosaic_output, mosaic_shape,fator)