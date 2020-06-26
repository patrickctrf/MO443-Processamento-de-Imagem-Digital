import os

import cv2
import numpy as np


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img


if __name__ == '__main__':
    """
Executing examples and generating report outputs.
    """
    # At first, create a directory for output images (if it does not exist yet).
    try:
        os.mkdir("output/")
    except OSError as error:
        print("O programa detectou que o diretório de saída já existe.",
              "O diretório foi criado com êxito.")

    # Mostra a imagem original a ser trabalhada neste roteiro
    cv2.imshow("Imagem Original", cv2.imread("bitmap.pbm", 0))
    cv2.waitKey(0)

    # Analisamos os pixels pretos, mas o OpenCV os considera valendo zero (ao 
    # contrario da notacao por nos adotada). Entao fazemos o complemento da 
    # imagem antes de qualquer procedimento.
    cv2.imwrite("bitmap-complemento.pbm", 255 - cv2.imread("bitmap.pbm", 0))

    # ==============ITEM-1======================================================
    imagem_dilatada_1 = cv2.dilate(cv2.imread("bitmap-complemento.pbm", 0), np.ones((1, 100), np.uint8))
    cv2.imshow("Item 1", imagem_dilatada_1)
    cv2.waitKey(0)
    cv2.imwrite("output/item1.pbm", imagem_dilatada_1)

    # ==============ITEM-2======================================================
    imagem_erodida_2 = cv2.erode(imagem_dilatada_1, np.ones((1, 100), np.uint8))
    cv2.imshow("Item 2", imagem_erodida_2)
    cv2.waitKey(0)
    cv2.imwrite("output/item2.pbm", imagem_erodida_2)

    # ==============ITEM-3======================================================
    imagem_dilatada_3 = cv2.dilate(cv2.imread("bitmap-complemento.pbm", 0), np.ones((200, 1), np.uint8))
    cv2.imshow("Item 3", imagem_dilatada_3)
    cv2.waitKey(0)
    cv2.imwrite("output/item3.pbm", imagem_dilatada_3)

    # ==============ITEM-4======================================================
    imagem_erodida_4 = cv2.erode(imagem_dilatada_3, np.ones((200, 1), np.uint8))
    cv2.imshow("Item 4", imagem_erodida_4)
    cv2.waitKey(0)
    cv2.imwrite("output/item4.pbm", imagem_erodida_4)

    # ==============ITEM-5======================================================
    imagem_and_5 = imagem_erodida_4 * imagem_erodida_2
    cv2.imshow("Item 5", imagem_and_5)
    cv2.waitKey(0)
    cv2.imwrite("output/item5.pbm", imagem_and_5)

    # ==============ITEM-6======================================================
    imagem_closing_6 = 255 * cv2.morphologyEx(imagem_and_5, cv2.MORPH_CLOSE, np.ones((1, 30), np.uint8))
    cv2.imshow("Item 6", imagem_closing_6)
    cv2.waitKey(0)
    cv2.imwrite("output/item6.pbm", imagem_closing_6)

    # ==============ITEM-7======================================================
    # Garante que a imagem eh binaria
    imagem_closing_7 = cv2.threshold(imagem_closing_6, 127, 255, cv2.THRESH_BINARY)[1]
    n_connected_components_7, imagem_connected_components_7 = cv2.connectedComponents(imagem_closing_7)
    imagem_colorida_componentes_7 = imshow_components(imagem_connected_components_7)
    cv2.imshow("Item 7", imagem_colorida_componentes_7)
    cv2.waitKey(0)
    cv2.imwrite("output/item7.png", imagem_colorida_componentes_7)
