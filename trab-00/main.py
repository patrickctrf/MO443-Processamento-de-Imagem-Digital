import cv2
import numpy as np
from skimage import exposure

# ==============Questao-1.1=====================================================

# Abrimos a imagem como escala de cinza. Queremos a matriz de representacao.
matrizImagem = cv2.imread("city.png", cv2.IMREAD_GRAYSCALE)

# item i: Complemento da imagem.
# A operacao de obter o complemento de um valor valor inteiro representado em
# binario corresponde apenas a substituir os bits "0" por "1" e vice-versa, que
# eh conhecido como "complemento de 1" e implementado pelo proprio OpenCV2.
matrizImagemComplementar = cv2.bitwise_not(matrizImagem)

# item ii: Contraste alterado.
# Esta instrução apenas faz uma interpolação dos valores na faixa de 0 a 255
# para a faixa de 100 a 200, implementada vetorialmente pela biblioteca numpy.
matrizImagemConstraste = np.interp(matrizImagem, [0, 255], [100, 200]).astype('uint8')

# item iii: Inverter linhas pares da matriz.
# Para tal, a funcao "range" gera o numero de cada linha da matriz de entrada
# e verificamos, atraves do modulo por 2, se a linha eh par ou impar. Quando eh
# par, a funcao "where" nos devolve a mesma linha da matriz, porem invertida.
# Quando eh impar, a linha original é devolvida. Assim, o processo se repete ate
# o final da construcao da matriz.
matrizImagemLinhasParesInvertidas = np.where(np.array(range(0, matrizImagem.shape[0])) % 2 == 0, np.flip(matrizImagem, 1), matrizImagem)

# item iv: Espelhar a matriz de imagem na metade (verticalmente).
# Para espelhar verticalmente, apenas recortamos a metade superior da matriz e
# concatenamos com ela mesma invertida (funcao "flip").
matrizImagemEspelhada = np.concatenate((matrizImagem[0:int(matrizImagem.shape[0] / 2)], np.flip(matrizImagem[0:int(matrizImagem.shape[0] / 2)], axis=0)), axis=0)

cv2.imshow("Original", matrizImagem)
cv2.waitKey(0)
cv2.imshow("Negativo", matrizImagemComplementar)
cv2.waitKey(0)
cv2.imshow("Contraste", matrizImagemConstraste)
cv2.waitKey(0)
cv2.imshow("Linhas Pares Invertidas", matrizImagemLinhasParesInvertidas)
cv2.waitKey(0)
cv2.imshow("Espelhada ao Meio Verticalmente", matrizImagemEspelhada)
cv2.waitKey(0)

# ==============FIM-da-Questao-1.1==============================================


# ==============Questao-1.2=====================================================

# Abrimos a imagem como escala de cinza. Queremos a matriz de representacao.
matrizImagem = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE)

# A funcao "adjust_gamma" exerce a correção de gamma proposta no roteiro, com a
# unica diferenca que o scikit-imagem considera gamma elevado a "-1", em relacao
# ao pedido no roteiro.
matrizImagemGamma1_5 = exposure.adjust_gamma(matrizImagem, gamma=1 / 1.5)
matrizImagemGamma2_5 = exposure.adjust_gamma(matrizImagem, gamma=1 / 2.5)
matrizImagemGamma3_5 = exposure.adjust_gamma(matrizImagem, gamma=1 / 3.5)

cv2.imshow("Gamma 1.0", matrizImagem)
cv2.waitKey(0)
cv2.imshow("Gamma 1.5", matrizImagemGamma1_5)
cv2.waitKey(0)
cv2.imshow("Gamma 2.5", matrizImagemGamma2_5)
cv2.waitKey(0)
cv2.imshow("Gamma 3.5", matrizImagemGamma3_5)
cv2.waitKey(0)

# ==============FIM-da-Questao-1.2==============================================


# ==============Questao-1.3=====================================================

# Abrimos a imagem como escala de cinza. Queremos a matriz de representacao.
matrizImagem = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE)

# Para o plano de cada bit, verificamos se o bit selecionado vale 0 ou 1
# aplicando uma mascara binaria na respectiva posicao binaria. Em seguida,
# normalizamos a matriz resultante: Cada elemento que possui o bit desejado em
# "1" eh dividido pelo bit de mesmo valor. Assim, a matriz se divide entre zeros
# e um, formando uma representacao binaria.
# O OpenCV2 mapeia automaticamente valores no intervalo [0,1] para o intervalo
# [0,255]. https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=imshow#cv2.imshow
cv2.imshow("Todos os planos", matrizImagem)
cv2.waitKey(0)
cv2.imshow("Plano de bit 0", (matrizImagem & 0b00000001) / 0b00000001)
cv2.waitKey(0)
cv2.imshow("Plano de bit 4", (matrizImagem & 0b00010000) / 0b00010000)
cv2.waitKey(0)
cv2.imshow("Plano de bit 7", (matrizImagem & 0b10000000) / 0b10000000)
cv2.waitKey(0)

# ==============FIM-da-Questao-1.3==============================================

# ==============Questao-1.4=====================================================

# Abrimos a imagem como escala de cinza. Queremos a matriz de representacao.
matrizImagem = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE)

matrizImagemMosaico = np.concatenate((
    np.concatenate((matrizImagem[128:256, 128:256], matrizImagem[256:384, 256:384], matrizImagem[384:512, 0:128], matrizImagem[0:128, 256:384]), axis=1),
    np.concatenate((matrizImagem[128:256, 384:512], matrizImagem[384:512, 384:512], matrizImagem[0:128, 0:128], matrizImagem[256:384, 0:128]), axis=1),
    np.concatenate((matrizImagem[256:384, 384:512], matrizImagem[384:512, 128:256], matrizImagem[0:128, 128:256], matrizImagem[128:256, 256:384]), axis=1),
    np.concatenate((matrizImagem[0:128, 384:512], matrizImagem[384:512, 256:384], matrizImagem[256:384, 128:256], matrizImagem[128:256, 0:128]), axis=1)),
    axis=0)

cv2.imshow("Nao Mosaico", matrizImagem)
cv2.waitKey(0)
cv2.imshow("Mosaico", matrizImagemMosaico)
cv2.waitKey(0)

# ==============FIM-da-Questao-1.4==============================================

# ==============Questao-1.5=====================================================

# Abrimos as imagens como escala de cinza. Queremos as matrizes de representacao.
babuinoImagem = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE)
borboletaImagem = cv2.imread("butterfly.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Babuino", babuinoImagem)
cv2.waitKey(0)
cv2.imshow("Borboleta", borboletaImagem)
cv2.waitKey(0)
cv2.imshow("Babuino02 + Borboleta08", (0.2 * babuinoImagem + 0.8 * borboletaImagem).astype('uint8'))
cv2.waitKey(0)
cv2.imshow("Babuino05 + Borboleta05", (0.5 * babuinoImagem + 0.5 * borboletaImagem).astype('uint8'))
cv2.waitKey(0)
cv2.imshow("Babuino08 + Borboleta02", (0.8 * babuinoImagem + 0.2 * borboletaImagem).astype('uint8'))
cv2.waitKey(0)

# ==============FIM-da-Questao-1.5==============================================
