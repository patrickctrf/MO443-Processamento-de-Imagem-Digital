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


def get_spectrum_from_img(img):
    """
Performs Fourier Fast Transform (FFT) over an given image matrix and returns its
fourier frequency spectrum.
    :param img: An image matrix to perform FFT.
    :return: The respective frequency spectrum.
    """
    # Executa FFT em 2 dimensoes
    f = np.fft.fft2(img)
    # Desloca a frequencia zero para o centro do espectro
    fshift = np.fft.fftshift(f)

    return fshift


def get_img_from_spectrum(spectrum):
    """
Performs Inverse Fourier Transform over a spectrum matrix and returns the
reconstructed image matrix.
    :param spectrum:
    :return: Reconstructed image.
    """
    # Retorna a frequencia zero colocada no centro do espectro de volta a
    # posicao original.
    f_ishift = np.fft.ifftshift(spectrum)
    # Realiza FFT inversa
    img_back = np.fft.ifft2(f_ishift)
    # Realiza a conversao para valores somente reais
    img_back = np.abs(img_back)

    return img_back


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

    # Abre a imagem original
    img = cv2.imread("baboon.png", 0)

    # Obtem o espectro de frequencias da imagem original, em valores complexos.
    spectrum = get_spectrum_from_img(img)
    # Extrai somente a magnitude retornada da FFT.
    magnitude_spectrum = 20 * np.log(np.abs(spectrum))

    # Reconstroi a imagem a partir do espectro resultante.
    img_reconstructed = get_img_from_spectrum(spectrum)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Imagem\n de \nEntrada'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude\n do \nEspectro'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_reconstructed, cmap='gray')
    plt.title('Imagem\n Reconstruida'), plt.xticks([]), plt.yticks([])
    plt.savefig("reconstrucao-da-imagem.png")
    plt.show()