import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


if __name__ == '__main__':
    img = cv2.imread("coins.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh = threshold_otsu(img_gray)
    # cross = cv2.morphologyEx(cross, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    # mask = cv2.morphologyEx(cross, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

    _, mask_coins = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    # _, mask_coins = cv2.threshold(img_gray, 184, 255, cv2.THRESH_BINARY)

    mask_coins = cv2.morphologyEx(mask_coins, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    mask_coins = cv2.morphologyEx(mask_coins, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    mask_coins = cv2.bitwise_not(mask_coins)
    plt.imshow(mask_coins, cmap='gray')
    plt.title('Filtrirana maska')
    plt.show()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    plt.imshow(sat, cmap='gray')
    plt.show()

    thresh2 = threshold_otsu(sat)

    _, copper_coin = cv2.threshold(sat, thresh2, 255, cv2.THRESH_BINARY)
    # _, copper_coin = cv2.threshold(sat, 47, 255, cv2.THRESH_BINARY)

    copper_coin = cv2.morphologyEx(copper_coin, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    plt.imshow(copper_coin, cmap='gray')
    plt.title('Copper coin')
    plt.show()

    marker = cv2.bitwise_and(copper_coin, mask_coins)

    reconstructed = morphological_reconstruction(marker, mask_coins)
    image = cv2.erode(reconstructed, np.ones((5, 5), np.uint8))  # erozija

    plt.title('Izdvojen bakarni novcic:')
    plt.imshow(image, cmap='gray')
    plt.show()
    cv2.imwrite("coin_mask.png", image)

    mask = np.zeros_like(img[:, :, 0])
    # selekcija svih redova i kolona, a 0 je za prvi (crveni)kanal, sve je ista slika samo s crvenim kanalom
    coin = cv2.bitwise_and(img, img, mask=image)
    plt.imshow(coin)
    plt.show()
