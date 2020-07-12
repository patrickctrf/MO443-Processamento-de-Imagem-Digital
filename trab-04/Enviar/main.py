import os

import cv2
import numpy as np


def draw_bounding_boxes(image, contornos):
    for i, c in enumerate(contornos):
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
        image = cv2.putText(image, str(i), (x - 20, y + h), 0, 0.3, (255, 255, 255))

    return image


def draw_text_non_text(image, contornos, text_lines_indexes):
    for i, c in enumerate(contornos):
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)

        if i in text_lines_indexes:
            image = cv2.putText(image, "TEXTO", (x - 40, y + h - 10), 0, 0.3, (255, 255, 255))
        else:
            image = cv2.putText(image, "NAO-TEXTO", (x - 60, y + h - 10), 0, 0.3, (255, 255, 255))

    return image


def draw_only_text(image, contornos, text_lines_indexes):
    for i, c in enumerate(contornos):
        if i in text_lines_indexes:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
            image = cv2.putText(image, " ", (x - 40, y + h - 10), 0, 0.3, (255, 255, 255))

    return image


def get_ratios(image, contornos):
    # Salvamos as porcentagens de cada parametro.
    list_black_percentiles = []
    list_transitions = []
    for i, c in enumerate(contornos):
        rect = cv2.boundingRect(c)
        # w = width, h = height.
        x, y, w, h = rect
        # Montamos o contorno da imagem de entrada em si
        img_snip = image[y:y + h, x:x + w]
        # Mais facil para a logica booleana da transicao falso-verdadeiro.
        # 255 vira 1, zero continua zero.
        img_snip = np.where(img_snip == 0, 0, 1)

        # Percentual entre pixels pretos e totais
        black_pixels = np.count_nonzero(img_snip)
        total_pixels = w * h
        list_black_percentiles.append(black_pixels / total_pixels)

        # Numero de transicoes horizontais entre pixels pretos e brancos
        indice_de_cada_transicao = np.where(np.roll(img_snip, shift=1, axis=1) < img_snip)[1]
        # Transicao no indice zero indica apenas que o ultimo pixels eh diferente do primeiro, nao queremos contar isso.
        indice_de_cada_transicao = np.delete(indice_de_cada_transicao, np.where(indice_de_cada_transicao == 0))
        n_transicoes_horizontais = len(indice_de_cada_transicao)

        # Numero de transicoes verticais entre pixels pretos e brancos
        indice_de_cada_transicao = np.where(np.roll(img_snip, shift=1, axis=0) < img_snip)[1]
        # Transicao no indice zero indica apenas que o ultimo pixels eh diferente do primeiro, nao queremos contar isso.
        indice_de_cada_transicao = np.delete(indice_de_cada_transicao, np.where(indice_de_cada_transicao == 0))
        n_transicoes_verticais = len(indice_de_cada_transicao)

        if black_pixels != 0:
            list_transitions.append((n_transicoes_verticais + n_transicoes_horizontais) / black_pixels)
        else:
            list_transitions.append(0)

    return list_black_percentiles, list_transitions


def text_identifier(list_black_percentiles, list_transitions):
    lbp = np.array(list_black_percentiles)
    ltr = np.array(list_transitions)

    return np.where((lbp < 0.98) & (lbp > 0.3) & (lbp > ltr) & (ltr < 0.4))[0]


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
    # cv2.imshow("Imagem Original", cv2.imread("bitmap.pbm", 0))
    # cv2.waitKey(0)

    # Analisamos os pixels pretos, mas o OpenCV os considera valendo zero (ao
    # contrario da notacao por nos adotada). Entao fazemos o complemento da
    # imagem antes de qualquer procedimento.
    cv2.imwrite("bitmap-complemento.pbm", 255 - cv2.imread("bitmap.pbm", 0))

    # ==============ITEM-1======================================================
    imagem_dilatada_1 = cv2.dilate(cv2.imread("bitmap-complemento.pbm", 0), np.ones((1, 100), np.uint8))
    # cv2.imshow("Item 1", imagem_dilatada_1)
    # cv2.waitKey(0)
    cv2.imwrite("output/item1.pbm", 255 - imagem_dilatada_1)

    # ==============ITEM-2======================================================
    imagem_erodida_2 = cv2.erode(imagem_dilatada_1, np.ones((1, 100), np.uint8))
    # cv2.imshow("Item 2", imagem_erodida_2)
    # cv2.waitKey(0)
    cv2.imwrite("output/item2.pbm", 255 - imagem_erodida_2)

    # ==============ITEM-3======================================================
    imagem_dilatada_3 = cv2.dilate(cv2.imread("bitmap-complemento.pbm", 0), np.ones((200, 1), np.uint8))
    # cv2.imshow("Item 3", imagem_dilatada_3)
    # cv2.waitKey(0)
    cv2.imwrite("output/item3.pbm", 255 - imagem_dilatada_3)

    # ==============ITEM-4======================================================
    imagem_erodida_4 = cv2.erode(imagem_dilatada_3, np.ones((200, 1), np.uint8))
    # cv2.imshow("Item 4", imagem_erodida_4)
    # cv2.waitKey(0)
    cv2.imwrite("output/item4.pbm", 255 - imagem_erodida_4)

    # ==============ITEM-5======================================================
    imagem_and_5 = (imagem_erodida_4 / 255 * imagem_erodida_2).astype(np.uint8)
    # cv2.imshow("Item 5", imagem_and_5)
    # cv2.waitKey(0)
    cv2.imwrite("output/item5.pbm", 255 - imagem_and_5)

    # ==============ITEM-6======================================================
    imagem_closing_6 = cv2.morphologyEx(imagem_and_5, cv2.MORPH_CLOSE, np.ones((1, 30), np.uint8))
    # cv2.imshow("Item 6", imagem_closing_6)
    # cv2.waitKey(0)
    cv2.imwrite("output/item6.pbm", 255 - imagem_closing_6)

    # ==============ITEM-7======================================================
    contornos, hierarquia = cv2.findContours(imagem_closing_6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagem_componentes_conexos_contornados_7 = imagem_closing_6.copy()
    imagem_componentes_conexos_contornados_7 = draw_bounding_boxes(imagem_componentes_conexos_contornados_7, contornos)

    n_contornos_item_7 = len(contornos)
    print("Encontrados ", n_contornos_item_7, " contornos.")
    # cv2.imshow("Item 7", imagem_componentes_conexos_contornados_7)
    # cv2.waitKey(0)
    cv2.imwrite("output/item7.pbm", 255 - imagem_componentes_conexos_contornados_7)

    # ==============ITEM-8======================================================
    contornos, hierarquia = cv2.findContours(imagem_closing_6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagem_componentes_conexos_contornados_8 = imagem_closing_6.copy()
    list_black_percentiles_item_8, list_transitions_item_8 = get_ratios(imagem_componentes_conexos_contornados_8, contornos)

    # ==============ITEM-9======================================================
    text_lines_indexes = text_identifier(list_black_percentiles_item_8, list_transitions_item_8)

    contornos, hierarquia = cv2.findContours(imagem_closing_6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagem_componentes_conexos_contornados_9a = imagem_closing_6.copy()
    imagem_componentes_conexos_contornados_9a = draw_text_non_text(imagem_componentes_conexos_contornados_9a, contornos, text_lines_indexes)

    print("Número de linhas na imagem: " + str(len(text_lines_indexes)))
    # cv2.imshow("Item 9 Componente Conexo", imagem_componentes_conexos_contornados_9a)
    # cv2.waitKey(0)
    cv2.imwrite("output/item9-connected-component.pbm", 255 - imagem_componentes_conexos_contornados_9a)

    contornos, hierarquia = cv2.findContours(imagem_closing_6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagem_componentes_conexos_contornados_9b = cv2.imread("bitmap-complemento.pbm", 0).copy()
    imagem_componentes_conexos_contornados_9b = draw_only_text(imagem_componentes_conexos_contornados_9b, contornos, text_lines_indexes)

    # cv2.imshow("Item 9 Imagem Original", imagem_componentes_conexos_contornados_9b)
    # cv2.waitKey(0)
    cv2.imwrite("output/item9-original-text.pbm", 255 - imagem_componentes_conexos_contornados_9b)

    # ==============ITEM-10=====================================================
    imagem_closing_10 = cv2.imread("bitmap-complemento.pbm", 0)
    for i in range(1):
        imagem_closing_10 = cv2.dilate(imagem_closing_10, np.ones((6, 2), np.uint8), iterations=1)
        imagem_closing_10 = cv2.morphologyEx(imagem_closing_10, cv2.MORPH_CLOSE, np.ones((6, 4), np.uint8), iterations=3)
        imagem_closing_10 = imagem_closing_10 / 255 * imagem_closing_6

    imagem_closing_10 = imagem_closing_10.astype(np.uint8)

    contornos, hierarquia = cv2.findContours(imagem_closing_10, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagem_componentes_contornados_10 = imagem_closing_10.copy()
    list_black_percentiles_item_10, list_transitions_item_10 = get_ratios(imagem_componentes_contornados_10, contornos)

    text_lines_indexes_10 = text_identifier(list_black_percentiles_item_10, list_transitions_item_10)

    imagem_closing_10_contornos = imagem_closing_10.copy()
    imagem_closing_10_contornos = draw_only_text(cv2.imread("bitmap-complemento.pbm", 0), contornos, text_lines_indexes_10)
    # cv2.imshow("Item 10", imagem_closing_10_contornos)
    # cv2.waitKey(0)
    cv2.imwrite("output/item10.pbm", 255 - imagem_closing_10_contornos)

    n_contornos_item_10 = len(text_lines_indexes_10)
    print("Total de blocos de palavras: ", n_contornos_item_10)

    # ==============TXT-OUTPUTS=================================================
    with open("output/list_black_percentiles.txt", "w") as file:
        for item in list_black_percentiles_item_8:
            file.write(str(item) + "\n")

    with open("output/list_transitions.txt", "w") as file:
        for item in list_transitions_item_8:
            file.write(str(item) + "\n")

    with open("output/n_contornos_item_7.txt", "w") as file:
        file.write(str(n_contornos_item_7) + "\n")

    with open("output/n_linhas_texto.txt", "w") as file:
        file.write(str(len(text_lines_indexes)) + "\n")

    with open("output/n_contornos_item_10.txt", "w") as file:
        file.write(str(n_contornos_item_10) + "\n")
