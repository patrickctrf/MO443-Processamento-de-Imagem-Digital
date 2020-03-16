import cv2
import numpy as np

#===============Questao-1.1=====================================================

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
matrizImagemConstraste = np.interp(matrizImagem, [0,255], [100,200]).astype('uint8')

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
matrizImagemEspelhada = np.concatenate((matrizImagem[0:int(matrizImagem.shape[0]/2)], np.flip(matrizImagem[0:int(matrizImagem.shape[0]/2)], axis=0)), axis=0)

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

#===============FIM-da-Questao-1.1==============================================

print(matrizImagem)