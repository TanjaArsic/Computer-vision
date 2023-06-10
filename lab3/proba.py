import cv2
import numpy as np


def merge_imgs(img1, img2):
    MIN_MATCH_COUNT = 10
    detector = cv2.SIFT_create()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = list()
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) <= MIN_MATCH_COUNT:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    h, w = img1.shape[0], img1.shape[1]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    dst += (w, 0)  # shift the transformed image to the right by the width of img1

    # Create a white canvas of size that fits both images
    h_new, w_new = max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]
    result = np.ones((h_new, w_new, 3), np.uint8)

    # Merge the images onto the canvas
    result[0:img1.shape[0], 0:img1.shape[1]] = img1
    warp_img = cv2.warpPerspective(img2, M, (w_new, h_new))
    result = np.where(warp_img != 0, warp_img, result)

    return result


if __name__ == '__main__':
    img1 = cv2.imread('img1.jpg')
    img2 = cv2.imread('img2.jpg')
    img3 = cv2.imread('img3.jpg')

    sift = cv2.SIFT_create()

    kp2, desc2 = sift.detectAndCompute(img2, None)
    kp3, desc3 = sift.detectAndCompute(img3, None)

    # 2. i 3. slika, FlannBased ne bruteforce
    flann_params = dict(algorithm=1, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    matches = matcher.match(desc2, desc3)

    # karakteristicne tacke iz obe slike
    pts2 = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts3 = np.float32([kp3[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # odredjivanje homografije
    H23, _ = cv2.findHomography(pts3, pts2, cv2.RANSAC)

    # Warp treca slika da se podudara s drugom slikom
    result12 = cv2.warpPerspective(img3, H23, (img2.shape[1] + img3.shape[1], int(1.2 * img2.shape[0])))
    # panorama
    result12[0:img2.shape[0], 0:img2.shape[1]] = img2

    cv2.imwrite('result12.jpg', result12)
    cv2.imshow('Result12', result12)

    # sad za panoramicu i prvu sliku isti postupak
    kp3, desc3 = sift.detectAndCompute(result12, None)
    kp1, desc1 = sift.detectAndCompute(img1, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc3, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # image = cv2.drawMatches(img1, kp1, result12, kp3, good, None, cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('uparivanje', image)

    matches = matcher.match(desc1, desc3)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts3 = np.float32([kp3[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H13, _ = cv2.findHomography(pts3, pts1, cv2.RANSAC)

    result = cv2.warpPerspective(result12, H13, (img1.shape[1] + result12.shape[1], int(1.2 * img1.shape[0])))
    result[0:img1.shape[0], 0:img1.shape[1]] = img1

    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
