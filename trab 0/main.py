import cv2
import numpy as np

#===============Questao-1.1=====================================================

# Abrimos a imagem como escala de cinza. Queremos a matriz de representacao.
matrizImagem = cv2.imread("city.png", cv2.IMREAD_GRAYSCALE)

# item i: Complemento da imagem.
matrizImagemComplementar = cv2.bitwise_not(matrizImagem)

# item ii: Contraste alterado.
matrizImagemConstraste = np.interp(matrizImagem, [0,255], [100,200]).astype('uint8')

# item iii: Inverter linhas pares da matriz.
matrizImagemLinhasParesInvertidas = np.where(np.array(range(0, matrizImagem.shape[0])) % 2 == 0, matrizImagem, np.flip(matrizImagem, 1))

# item iv: Espelhar a matriz de imagem na metade (verticalmente).
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

print(matrizImagem)