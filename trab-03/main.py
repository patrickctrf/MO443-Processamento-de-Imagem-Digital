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
    # Como o raio do circulo nao inclui o ponto central, precisamos definir que
    # aquele ponto vale zero se o proprio raio for zero.
    if r == 0:
        return np.zeros((N, N))

    # Comprimento do raio em cada dimensao
    rx = ry = N / 2
    # Fazemos um preenchimento linear em cada dimensao para nos auxiliar a
    # calcular a distancia do centro da matriz
    x, y = np.indices((N, N))

    # Fazemos um simples calculo de hipotenusa para comparar a distancia do
    # centro da matriz a cada pixel (raio do circulo em relacao ao raio pretendido)
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


def apply_fourier_filter(spectrum, min_db=0, max_db=512):
    """
Dada uma matriz de espectro de entrada, aplica filtro de frequencias 2d deixando
passar frequencias acima do valor minimo e abaixo do valor maximo.
    :param spectrum: Matriz de espectro com todas as frequencias originais (pode
    ser espectro completo ou apenas a magnitude).
    :param min_db: Frequencia minima a ser passada.
    :param max_db: Frequencia maxima a ser passada.
    :return: A matriz apos a aplicacao do filtro.
    """
    # Aqui decidimos se vamos fazer um circulo de numeros "1" cercado por uma
    # matriz de zeros, ou um circulo de "zeros" cercado por um matriz de "1".
    if max_db > min_db:
        return spectrum * (circle_into_matrix(r=max_db, N=512) - circle_into_matrix(r=min_db, N=512))
    else:
        filtro_complementar = circle_into_matrix(r=max_db, N=512) - circle_into_matrix(r=min_db, N=512)
        return spectrum * (np.where(filtro_complementar == 0, 1, 0))


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

    # =======CALCULO-DE-ESPECTRO-E-RECONSTRUCAO-DA-IMAGEM-ORIGINAL==============

    # Abre a imagem original
    img = cv2.imread("baboon.png", 0)

    # Obtem o espectro de frequencias da imagem original, em valores complexos.
    spectrum = get_spectrum_from_img(img)
    # Extrai somente a magnitude retornada da FFT.
    magnitude_spectrum = 20 * np.log(np.abs(spectrum))

    # Reconstroi a imagem a partir do espectro resultante.
    img_reconstructed = get_img_from_spectrum(spectrum)

    # Plotando as imagens obtidas.
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Imagem\n de \nEntrada'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude\n do \nEspectro'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_reconstructed, cmap='gray')
    plt.title('Imagem\n Reconstruida'), plt.xticks([]), plt.yticks([])
    plt.savefig("reconstrucao-da-imagem.png")
    plt.show()

    # ==========MONTAGEM-DOS-FILTROS-E-FILTRAGEM================================

    nucleo_passa_baixa = apply_fourier_filter(spectrum, 0, 80)
    nucleo_passa_alta = apply_fourier_filter(spectrum, 80, 0)
    nucleo_passa_faixa = apply_fourier_filter(spectrum, 40, 80)

    # Plotando as imagens obtidas.
    plt.subplot(131), plt.imshow(20 * np.ma.log(np.abs(nucleo_passa_baixa)).filled(0), cmap='gray')
    plt.title('Nucleo\nPassa\nBaixa'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(20 * np.ma.log(np.abs(nucleo_passa_alta)).filled(0), cmap='gray')
    plt.title('Nucleo\nPassa\nAlta'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(20 * np.ma.log(np.abs(nucleo_passa_faixa)).filled(0), cmap='gray')
    plt.title('Nucleo\nPassa\nFaixa'), plt.xticks([]), plt.yticks([])
    plt.savefig("nucleos.png")
    plt.show()

    img_reconstructed_fpb = get_img_from_spectrum(nucleo_passa_baixa)
    img_reconstructed_fpa = get_img_from_spectrum(nucleo_passa_alta)
    img_reconstructed_fpf = get_img_from_spectrum(nucleo_passa_faixa)

    # Plotando as imagens obtidas.
    plt.subplot(131), plt.imshow(img_reconstructed_fpb, cmap='gray')
    plt.title('Filtragem\nPassa\nBaixa'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_reconstructed_fpa, cmap='gray')
    plt.title('Filtragem\nPassa\nAlta'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_reconstructed_fpf, cmap='gray')
    plt.title('Filtragem\nPassa\nFaixa'), plt.xticks([]), plt.yticks([])
    plt.savefig("filtragem.png")
    plt.show()

    # ============COMPRESSAO====================================================

    max_spectrum = magnitude_spectrum.max()
    compressed_spectrum = np.where(magnitude_spectrum < 0.7 * max_spectrum, 0, spectrum)

    img_reconstructed_compressed = get_img_from_spectrum(compressed_spectrum)

    # Plotando as imagens obtidas.
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Imagem\n de \nEntrada'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_reconstructed_compressed, cmap='gray')
    plt.title('Imagem\n Reconstruida \nCompressao'), plt.xticks([]), plt.yticks([])
    plt.savefig("reconstrucao-da-imagem-compressed.png")
    plt.show()
