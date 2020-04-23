import os

import cv2
from scipy.ndimage import convolve, rotate
from numpy import array, outer, transpose, ones, sqrt


def convolui(original_file_path, filtro, output_file_path="convolve-img.png"):
    """
Recebe uma imagem uint8 em escala de cinza, um filtro matricial e a retorna, 
salvando tambem no arquivo de saida especificado.
    :param original_file_path: A imagem original
    :param filtro: O filtro a se aplciar
    :param output_file_path: Path da imagem de saida
    :return: Matriz da imagem de saida
    """
    img = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)

    # Utilizamos a operacao de convolucao implementada eficientemente no pacote
    # scipy.
    output_img = convolve(img, filtro).round().astype("uint8")

    cv2.imwrite(output_file_path, output_img)

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
           [-1, -2, -16, -2, -1],
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
    convolui(original_file_path="baboon.png", filtro=h_3, output_file_path="output/mask_h3_result.png")

    # Filtro/mascara que iremos aplicar.
    h_4 = transpose(h_3)
    convolui(original_file_path="baboon.png", filtro=h_4, output_file_path="output/mask_h4_result.png")

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

    # Filtro/mascara que iremos aplicar.
    h_34 = sqrt(h_3 ** 2 + h_4 ** 2)
    convolui(original_file_path="baboon.png", filtro=h_34, output_file_path="output/mask_h34_result.png")


if __name__ == '__main__':
    main()
