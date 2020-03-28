import cv2
import numpy as np


def change_resolution(original_file_path, output_file_path="resized-img.png", scale_percent=100):
    """
Receives an grayscale image path and saves it back with new resolution scale.
    :param original_file_path: Original graysacale uint8 image.
    :param output_file_path: Rescaled output graysacale uint8 image.
    :param scale_percent: Reduction rate
    :return: 0 if success.
    """

    img = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_file_path, resized_img)
    cv2.destroyAllWindows()

    # If everything has gone fine
    return 0


def change_deepth(original_file_path, output_file_path="deepth-img.png", original_deepth=256, new_deepth=256):
    """
Changes the deepth for a given grayscale uint8 image.
    :param original_file_path: Original graysacale uint8 image.
    :param output_file_path: New deepth output graysacale uint8 image.
    :param original_deepth: Original image deepth.
    :param new_deepth: New image deepth
    :return: 0 if success.
    """

    img = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)

    deepth_img = np.around(np.around((new_deepth - 1) / (original_deepth - 1) * img) / (new_deepth - 1) * (256 - 1)).astype("uint8")

    cv2.imwrite(output_file_path, deepth_img)
    cv2.destroyAllWindows()

    # If everything has gone fine
    return 0


def change_contrast(original_file_path, output_file_path="constrast-img.png", operation=None, alpha=2, beta=2, gamma=2):
    """
Manipulates constrast on grayscale uint8 image.
    :param original_file_path: Original graysacale uint8 image.
    :param output_file_path: New deepth output graysacale uint8 image.
    :param operation: Which operation to perform as string (log, exp, quad, sqrt or sigma).
    :param alpha: alpha factor. Default is 2.
    :param beta: beta factor. Default is 2.
    :param gamma: gamma factor. Default is 2.
    :return:  0 if success.
    """

    # Random constant values
    a = 255 // 3
    b = 2 * 255 // 3
    c = 1

    img = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)

    # Allocating a bit more memory for our operations. Must cast back to uint8
    # before finishing.
    img = img.astype("float64")

    # Switch case about witch operation user selected.
    if operation == "log":
        img = c * np.log(img + 1)

    if operation == "exp":
        img = c * np.exp(img)

    if operation == "quad":
        img = c * np.power(img, 2)

    if operation == "sqrt":
        img = c * np.power(img, 1 / 2)

    if operation == "sigma":
        if img.max() < a:
            img *= alpha

        elif img.max() < b:
            img = beta * (img - a) + alpha * a

        else:
            img = gamma * (img - b) + beta * (b - a) + alpha * a

    # Some operations may give bad range values for visualization.
    # I'm normalizing image to range from 0 to 255 again.
    img = img * 255 / img.max()

    cv2.imwrite(output_file_path, img.round().astype("uint8"))
    cv2.destroyAllWindows()

    # If everything has gone fine
    return 0


def main():
    """
Just executing examples and generating report outputs.
    """

    # 1.1: Resolucao de imagens
    change_resolution("baboon.png", scale_percent=10)

    # 1.2: Profundidade de imagens
    change_deepth("baboon.png", new_deepth=2)

    # 1.3: Manipulacao de contraste
    change_contrast("baboon.png", operation="log")


if __name__ == '__main__':
    main()
