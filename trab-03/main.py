import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def circle_into_matrix(r, N=512):
    """
Retorna uma matriz NxN com um circulo de raio "r" alinhado no meio da matriz e
formado por elementos "1". Os elementos fora do raio do circulo valem "0".
    :param r: Raio do circulo centrado na matriz.
    :param N: Tamanho da matriz a ser retornada.
    :return: Matriz NxN com um circulo de raio "r" alinhado no meio da matriz e
    formado por elementos "1". Os elementos fora do raio do circulo valem "0".
    """

    if r == 0:
        return np.zeros((N, N))

    rx = ry = N / 2
    x, y = np.indices((N, N))
    return (np.hypot(rx - x, ry - y) - r < 0.1).astype(int)


if __name__ == '__main__':
    """
Executing examples and generating report outputs.
    """

    # At first, create a directory for output images (if it does not exist yet).
    try:
        os.mkdir("output/")
    except OSError as error:
        print("O programa detectou que o diretório de saída já existe.",
              "As imagens foram sobrescritas, mas o programa foi executado com êxito.")

    mtx_result = circle_into_matrix(0, N=20)

    img = cv2.imread('baboon.png', 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
