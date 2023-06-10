import cv2
import numpy as np


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


if __name__ == '__main__':
    img = cv2.imread('slika2.jpg', cv2.IMREAD_GRAYSCALE)  # slika1.jpg
    cv2.imshow('Original image:', img)

    h, w = img.shape[:2]
    marker = np.zeros((h, w), dtype=np.uint8)

    marker[:, 0] = img[:, 0]  # levi border
    marker[:, -1] = img[:, -1]  # desni border
    marker[0, :] = img[0, :]  # gornji border
    marker[-1, :] = img[-1, :]  # donji border
    # cv2.imshow('Marker', marker)

    marker = morphological_reconstruction(marker, img)
    mask = cv2.bitwise_not(marker)
    out = cv2.bitwise_and(img, mask)

    cv2.imwrite('filtered.jpg', out)
    cv2.imshow('Filtered image:', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
