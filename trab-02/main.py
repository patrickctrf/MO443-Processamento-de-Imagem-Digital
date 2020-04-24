import os

import cv2
from scipy.ndimage import convolve, rotate, correlate
from numpy import array, outer, transpose, ones, sqrt, interp, where


def convolui(original_file_path, filtro, output_file_path="convolve-img.png"):
    """
Recebe uma imagem uint8 em escala de cinza, um filtro matricial e a retorna,
salvando tambem no arquivo de saida especificado.
    :param original_file_path: A imagem original
    :param filtro: O filtro a se aplciar
    :param output_file_path: Path da imagem de saida
    :return: Matriz da imagem de saida
    """

    # Exibe a matriz do filtro sendo usado na saida padrao
    print("filtro: \n", filtro)

    img = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)

    # Convertemos os elementos para float64 apenas para garantir a precisao dos
    # calculos, convertendo-os de volta para uint8 ao retornar a imagem.
    img = img.astype("float64")

    # Utilizamos a operacao de convolucao implementada eficientemente no pacote
    # scipy.
    output_img = convolve(img, filtro)

    # Valores que extrapolem os limites inferior e superior de cada pixel sao
    # truncados.
    output_img = where(output_img < 0, 0, output_img)
    output_img = where(output_img > 255, 255, output_img)

    # Fazemos uma interpolacao so para garantir que esta tudo dentro da faixa
    # pretendida de valores: from [min, max] to [0, 255].
    output_img_min, output_img_max = output_img.min(), output_img.max()
    output_img = interp(output_img, array([output_img_min, output_img_max]), array([0.0, 255.0])).round().astype("uint8")

    cv2.imwrite(output_file_path, output_img)
    cv2.destroyAllWindows()

    # Exibe na saida padrao a matriz da imagem sendo retornada.
    print("output_img: \n", output_img)

    return output_img


def main():
    """
Executing examples and generating report outputs.
    """

    # At first, create a directory for output images (if it does not exist yet).
    try:
        os.mkdir("output/")
    except OSError as error:
        print(error)
        print("O programa detectou que o diretório de saída já existe.",
              "As imagens foram sobrescritas, mas o programa foi executado com êxito.")

    # Filtro/mascara que iremos aplicar.
    h_1 = [[0, 0, -1, 0, 0],
           [0, -1, -2, -1, 0],
           [-1, -2, 16, -2, -1],
           [0, -1, -2, -1, 0],
           [0, 0, -1, 0, 0]]
    h_1 = 1 * array(h_1)
    convolui(original_file_path="baboon.png", filtro=h_1, output_file_path="output/mask_h1_result.png")

    # Filtro/mascara que iremos aplicar.
    h_2 = array([1, 4, 6, 4, 1])
    h_2 = 1 / 256 * outer(h_2, h_2)
    convolui(original_file_path="baboon.png", filtro=h_2, output_file_path="output/mask_h2_result.png")

    # Filtro/mascara que iremos aplicar.
    h_3 = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]
    h_3 = 1 * array(h_3)
    img3 = convolui(original_file_path="baboon.png", filtro=h_3, output_file_path="output/mask_h3_result.png")

    # Filtro/mascara que iremos aplicar.
    h_4 = transpose(h_3)
    img4 = convolui(original_file_path="baboon.png", filtro=h_4, output_file_path="output/mask_h4_result.png")

    # Filtro/mascara que iremos aplicar.
    h_5 = [[-1, -1, -1],
           [-1, 8, -1],
           [-1, -1, -1]]
    h_5 = 1 * array(h_5)
    convolui(original_file_path="baboon.png", filtro=h_5, output_file_path="output/mask_h5_result.png")

    # Filtro/mascara que iremos aplicar.
    h_6 = 1 / 9 * ones((3, 3))
    convolui(original_file_path="baboon.png", filtro=h_6, output_file_path="output/mask_h6_result.png")

    # Filtro/mascara que iremos aplicar.
    h_7 = [[-1, -1, 2],
           [-1, 2, -1],
           [2, -1, -1]]
    h_7 = 1 * array(h_7)
    convolui(original_file_path="baboon.png", filtro=h_7, output_file_path="output/mask_h7_result.png")

    # Filtro/mascara que iremos aplicar.
    h_8 = rotate(h_7, angle=90)
    convolui(original_file_path="baboon.png", filtro=h_8, output_file_path="output/mask_h8_result.png")

    # ========Filtro-de-Sobel===================================================
    # ========h3-e-h4-combinados================================================

    # Imagem combinada dos filtros 3 e 4 eh obtida a partir das saidas das 
    # proprias outras.
    img34 = sqrt(img3.astype("float64") ** 2 + img4.astype("float64") ** 2)

    # Valores que extrapolem os limites inferior e superior de cada pixel sao
    # truncados.
    img34 = where(img34 < 0, 0, img34)
    img34 = where(img34 > 255, 255, img34)

    # Fazemos uma interpolacao so para garantir que esta tudo dentro da faixa
    # pretendida de valores: from [min, max] to [0, 255].
    img34_min, img34_max = img34.min(), img34.max()
    img34 = interp(img34, array([img34_min, img34_max]), array([0.0, 255.0])).round().astype("uint8")

    cv2.imwrite("output/mask_h34_result.png", img34)
    cv2.destroyAllWindows()

    # ========fim-Filtro-de-Sobel===============================================
    # ========h3-e-h4-combinados================================================


if __name__ == '__main__':
    main()
