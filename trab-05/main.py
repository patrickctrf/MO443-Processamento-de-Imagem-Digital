import os

import cv2


def open_images_grayscale(n_images):
    """
Esta função é específica para este roteiro e abre todas as N imagens selecionadas no argumento e retorna em listas como escala de cinza.

    :param n_images: Quantas das imagens devem ser abertas (em ordem).
    :return: Duas listas, a primeira com as imagens A e a segunda com imagens B (em ordem).
    """
    images_a = []
    images_b = []

    for i in range(n_images):
        images_a.append(cv2.cvtColor(cv2.imread("foto" + str(i + 1) + "A.jpg"), cv2.COLOR_BGR2GRAY))
        images_b.append(cv2.cvtColor(cv2.imread("foto" + str(i + 1) + "B.jpg"), cv2.COLOR_BGR2GRAY))

    return images_a, images_b


def keypoints_and_descriptors(img1, method):
    """
Recebe uma imagem do cv2 e um método de detector como argumentos. Encontra descritores e pontos de interesse.

    :param img1: Imagem a ser analisada.
    :param method: Método de análise.
    :return: Imagem com os pontos encontrados, os pontos de interesse e os descritores.
    """
    img = 0

    if method == "sift":
        # https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
        detector = cv2.xfeatures2d.SIFT_create()

    elif method == 'surf':
        # https://docs.opencv.org/master/df/dd2/tutorial_py_surf_intro.html
        detector = cv2.xfeatures2d.SURF_create(400)

    elif method == 'brief':
        # https://docs.opencv.org/3.4/dc/d7d/tutorial_py_brief.html
        star = cv2.xfeatures2d.StarDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp = star.detect(img1, None)
        keypoints, descriptors = brief.compute(img1, kp)
        img = cv2.drawKeypoints(img1, keypoints, img)

        return img, keypoints, descriptors

    else:  # ORB
        # https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
        orb = cv2.ORB_create()
        keypoints = orb.detect(img1, None)
        keypoints, descriptors = orb.compute(img1, keypoints)
        img = cv2.drawKeypoints(img1, keypoints, img)

        return img, keypoints, descriptors

    keypoints, descriptors = detector.detectAndCompute(images_a[0], None)
    img = cv2.drawKeypoints(img1, keypoints, img)

    return img, keypoints, descriptors


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

    # ==============ITEM-1======================================================
    images_a, images_b = open_images_grayscale(5)

    # ==============ITEM-2======================================================

    img, _, _ = keypoints_and_descriptors(images_a[0], method="sift")
    cv2.imshow('Pontos de interesse', img)
    cv2.waitKey(0)
    cv2.imwrite("output/item2_sift.jpg", img)

    img, _, _ = keypoints_and_descriptors(images_a[0], method="surf")
    cv2.imshow('Pontos de interesse', img)
    cv2.waitKey(0)
    cv2.imwrite("output/item2_surf.jpg", img)

    img, _, _ = keypoints_and_descriptors(images_a[0], method="brief")
    cv2.imshow('Pontos de interesse', img)
    cv2.waitKey(0)
    cv2.imwrite("output/item2_brief.jpg", img)

    img, _, _ = keypoints_and_descriptors(images_a[0], method="orb")
    cv2.imshow('Pontos de interesse', img)
    cv2.waitKey(0)
    cv2.imwrite("output/item2_orb.jpg", img)
