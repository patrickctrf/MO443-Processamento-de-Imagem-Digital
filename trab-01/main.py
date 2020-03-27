import cv2


def change_resolution(original_file_path, resized_file_path="resized-img.png", scale_percent=1):
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


def main():
    change_resolution("baboon.png", scale_percent=10)


if __name__ == '__main__':
    main()
