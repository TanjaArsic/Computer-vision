import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def fft(img):
    # img_fft = np.fft.fft2(img)  # pretvaranje slike iz spacijalnog u frekventni domen (FFT - Fast Fourier
    # Transform), fft2 je jer je u 2 dimenzije
    img_fft = np.fft.fftshift(np.fft.fft2(img))  # pomeranje koordinatnog pocetka u centar slike
    return img_fft


def inverse_fft(magnitude_log, complex_moduo_1):
    img_fft = complex_moduo_1 * np.exp(
        magnitude_log)  # vracamo amplitudu(magn) iz logaritma i mnozimo sa kompleksnim brojevima na slici
    img_filtered = np.abs(np.fft.ifft2(
        img_fft))  # funkcija ifft2 vraca sliku iz frekventnog u prostorni domen, nije potrebno raditi ifftshift jer
    # to se implicitno izvrsava
    # rezultat ifft2 je opet kompleksna slika, ali nas zanima samo moduo jer to je nasa slika zato opet treba np.abs()

    return img_filtered


def find_noise(img, threshold):
    height, width = img.shape
    cntr = find_image_center(img)
    radius = 10  # empirijski
    # avg = np.mean(img)
    # print(avg)
    for x in range(width):
        for y in range(height):
            if (x - cntr[0]) ** 2 + (y - cntr[1]) ** 2 >= radius ** 2:
                # x != cntr[0] or y != cntr[1]):
                if img[x, y] > threshold:
                    img[x, y] = np.mean(img)  # 0
    return img


def fix_noise(img, center, offset):
    img_fft = fft(img)  # prevodi se u frekv. domen
    img_fft_mag = np.abs(
        img_fft)  # slika u frekventnom domenu je kompleksan broj, nama je potrebna amplituda tog kompleksnog broja (
    # odnosno moduo sto daje funkcija np.abs())
    # koristi se da normalizuje fft vrednosti kasnije
    img_mag_1 = img_fft / img_fft_mag  # cuvanje kompleksnih brojeva sa jedinicnim moduom, jer cemo da menjamo amplitudu
    # kompleksni/amplituda = normalizuje, koristi se da se izbegne pojacanje suma kod filtriranja
    img_fft_log = np.log(img_fft_mag)  # vrednosti su prevelike da bi se menjale direktno
    # cisto da moze da se vizualizuje amplituda frekventnog domena

    # img_fft_log[center[0] - offset, center[1] - offset] = 0
    # img_fft_log[center[0] + offset, center[1] - offset] = 10  # ovo postavlja sum ali ispravlja strcanje
    # img_fft_log[center[0] - offset, center[1] + offset] = 10  # 10 se gotovo ne vidi, al 0 je crno pa taman
    # img_fft_log[center[0] + offset, center[1] + offset] = 0  # ovo je s cs
    # print(img_fft_log[206, 206])

    img_fft_log = find_noise(img_fft_log, 14.413)  # 14.414054896887066

    img_filtered = inverse_fft(img_fft_log, img_mag_1)  # vracanje u spacijalni domen da se isfiltrira slika

    return img_filtered


def extract_magnitude(img):  # racuna amplitudu koristeci diskretnu FT i vraca kao matricu, 2D niz koji predstavlja
    # frekventni sadrzaj slike
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # dft se primenjuje na sliku
    # s tim da se ovaj drugi argument odnosi na to da izlaz treba bude kompleksni array s 2 kanala, jedan realni i
    # jedan imaginarni deo
    dft_shift = np.fft.fftshift(dft)  # shiftuje se u centar komponenta s 0 frekvn. classic
    magnitude_spectrum = 20 * np.log(cv2.magnitude(  # magnitude racuna amplitudu kompleksnog arraya na lokaciji
        # svakog piksela, a 20*log prevodi u decibele
        dft_shift[:, :, 0],  # indeksiranje izdvaja sve vrste i kolone prvog, realnog kanala
        dft_shift[:, :, 1])  # indeksiranje izdvaja sve vrste i kolone drugog, imaginarnog kanala
    )
    return magnitude_spectrum


def extract_magnitude2(img):
    img_fft = fft(img)
    magnitude_spectrum = 20 * np.log(np.abs(img_fft))
    return magnitude_spectrum


def find_image_center(image):
    height, width = image.shape[:2]  # (height, width) za grayscale jer nema treci parametar koji predstavlja kanale
    center_x = int(width / 2)
    center_y = int(height / 2)
    return center_x, center_y


if __name__ == '__main__':
    img = cv2.imread("input.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.title(f'Slika (broj 4) sa šumom:')
    plt.imshow(img, cmap='gray')
    plt.show()

    center = find_image_center(img)  # center = (256, 256)

    fft_mag = extract_magnitude2(img)
    plt.title(f'Slika amplitude spektra sa tačkama koje generišu šum:')
    plt.imshow(fft_mag)  # sareno
    plt.show()
    cv2.imwrite("fft_mag.png", fft_mag)

    filtered = fix_noise(img, center, 50)
    fft_mag_filtered = extract_magnitude2(filtered)
    plt.title(f'Slika sa uklonjenim tačkama koje su generisale šum:')
    plt.imshow(fft_mag_filtered)
    plt.show()
    cv2.imwrite("fft_mag_filtered.png", fft_mag_filtered)

    plt.title(f'Slika sa uklonjenim šumom:')
    plt.imshow(filtered, cmap='gray')
    plt.show()
    cv2.imwrite("output.png", filtered)
