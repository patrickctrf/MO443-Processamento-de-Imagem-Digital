import cv2
import numpy as np


def change_resolution(original_file_path, resized_file_path="resized-img.png", scale_percent=100):
    """
Receives an grayscale image path and saves it back with new resolution scale.
    :param original_file_path: The original file
    :param resized_file_path:  The rescaled file output.
    :param scale_percent: Reduction rate
    :return: 0 if success, error string if error.
    """

    try:
        img = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)

        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        cv2.imwrite(resized_file_path, resized_img)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Exception trying to resize the image: ", e)
        return e

    # If everything is gone fine
    return 0


def change_deepth(original_file_path, new_deepth_file_path="deepth-img.png", original_deepth=256, new_deepth=256):
    try:
        img = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)

        deepth_img = np.around(np.around((new_deepth - 1) / (original_deepth - 1) * img) / (new_deepth - 1) * (256 - 1)).astype("uint8")

        cv2.imwrite(new_deepth_file_path, deepth_img)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Exception trying to image deepth: ", e)
        return e

    # If everything is gone fine
    return 0


def main():
    # 1.1: Resolucao de imagens
    change_resolution("baboon.png", scale_percent=10)

    # 1.2: Profundidade de imagens
    change_deepth("baboon.png", new_deepth=2)


if __name__ == '__main__':
    main()
