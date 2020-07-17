import os

import cv2
from numpy import array


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


def keypoints_and_descriptors(img1, method=None):
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
        detector = cv2.ORB_create()

    keypoints, descriptors = detector.detectAndCompute(img1, None)
    img = cv2.drawKeypoints(img1, keypoints, img)

    return img, keypoints, descriptors


def select_best_matches(matches, limiar=0.2):

    good_matches = []
    max_distance = 0

    for single_match in matches:
        if single_match.distance > max_distance:
            max_distance = single_match.distance

    for single_match in matches:
        if single_match.distance/max_distance < limiar:
            good_matches.append(single_match)

    return good_matches


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
    metodos_a_testar = ["sift", "surf", "brief", "orb"]
    img_kp_des_a = []
    img_kp_des_b = []

    for i, (img_a, img_b) in enumerate(zip(images_a, images_b)):
        for metodo in metodos_a_testar:
            img1, keypoints1, descriptors1 = keypoints_and_descriptors(img_a, method=metodo)
            # cv2.imshow('Pontos de interesse ' + str(metodo), img1)
            # cv2.waitKey(0)
            cv2.imwrite("output/item2_A_foto" + str(i + 1) + str(metodo) + ".jpg", img1)
            img_kp_des_a.append((img1, keypoints1, descriptors1))

            img2, keypoints2, descriptors2 = keypoints_and_descriptors(img_b, method=metodo)
            # cv2.imshow('Pontos de interesse ' + str(metodo), img2)
            # cv2.waitKey(0)
            cv2.imwrite("output/item2_B_foto" + str(i + 1) + str(metodo) + ".jpg", img2)
            img_kp_des_b.append((img2, keypoints2, descriptors2))

    # ==============ITEM-3======================================================
    metodos_a_testar = ["sift", "surf", "brief", "orb"]
    distances = []

    for i, (img_a, img_b) in enumerate(zip(images_a, images_b)):
        for metodo in metodos_a_testar:
            img1, keypoints1, descriptors1 = keypoints_and_descriptors(img_a, method=metodo)
            img2, keypoints2, descriptors2 = keypoints_and_descriptors(img_b, method=metodo)

            if metodo == "sift" or metodo == "surf":
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

            if metodo == "orb" or metodo == "brief":
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            if metodo == "sift":
                matches = select_best_matches(matches, limiar=0.1)

            if metodo == "brief":
                matches = select_best_matches(matches, limiar=0.3)

            if metodo == "orb":
                matches = select_best_matches(matches, limiar=0.2)

            if metodo == "surf":
                matches = select_best_matches(matches, limiar=0.2)

            img = cv2.drawMatches(img_a, keypoints1, img_b, keypoints2, matches[:],
                                  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # cv2.imshow('Distancias ' + str(metodo), img)
            # cv2.waitKey(0)
            cv2.imwrite("output/item3_foto" + str(i + 1) + str(metodo) + ".jpg", img)
