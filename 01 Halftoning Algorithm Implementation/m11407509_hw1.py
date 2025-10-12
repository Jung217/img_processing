import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 生成 Bayer matrix
def generate_bayer_matrix(n):
    if n == 1:
        return np.array([[0, 2], [3, 1]])
    else:
        smaller_matrix = generate_bayer_matrix(n - 1)
        size = 2 ** n
        new_matrix = np.zeros((size, size), dtype=int)
        for i in range(2 ** (n - 1)):
            for j in range(2 ** (n - 1)):
                base_value = 4 * smaller_matrix[i, j]
                new_matrix[i, j] = base_value
                new_matrix[i, j + 2 ** (n - 1)] = base_value + 2
                new_matrix[i + 2 ** (n - 1), j] = base_value + 3
                new_matrix[i + 2 ** (n - 1), j + 2 ** (n - 1)] = base_value + 1
        return new_matrix

def generate_thresholds_matrix(bayer_matrix):
    N = bayer_matrix.shape[0]
    thresholds_matrix = 255 * (bayer_matrix + 0.5) / (N * N)
    return thresholds_matrix

def Ordered_Dithering(img, thresholds_matrix):
    N = thresholds_matrix.shape[0]
    h, w = img.shape
    Ordered_Dithering_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            threshold = thresholds_matrix[i % N, j % N]
            Ordered_Dithering_img[i, j] = 255 if img[i, j] > threshold else 0
    return Ordered_Dithering_img

def Error_Diffusion(img):
    img = img.astype(float)
    h, w = img.shape
    Error_Diffusion_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            old_pixel = img[i, j]
            new_pixel = 255 if old_pixel > 127 else 0
            Error_Diffusion_img[i, j] = new_pixel
            quant_error = old_pixel - new_pixel
            if j + 1 < w:
                img[i, j+1] += quant_error * 7/16
            if i + 1 < h:
                if j > 0:
                    img[i+1, j-1] += quant_error * 3/16
                img[i+1, j] += quant_error * 5/16
                if j + 1 < w:
                    img[i+1, j+1] += quant_error * 1/16
    return Error_Diffusion_img.astype(np.uint8)

def hpsnr(original, halftoned):
    mse = np.mean((original.astype(float) - halftoned.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)

if __name__ == '__main__':
    img = cv.imread('images/Baboon-image.png', cv.IMREAD_GRAYSCALE)
    n = 2
    bayer_matrix = generate_bayer_matrix(n)
    thresholds_matrix = generate_thresholds_matrix(bayer_matrix)

    ordered_img = Ordered_Dithering(img, thresholds_matrix)
    error_diff_img = Error_Diffusion(img)

    # show
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title('Ordered Dithering')
    plt.imshow(ordered_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title('Error Diffusion')
    plt.imshow(error_diff_img, cmap='gray')
    plt.axis('off')
    plt.show()

    cv.imwrite('result/baboon_ordered_dithering.png', ordered_img)
    cv.imwrite('result/baboon_error_diffusion.png', error_diff_img)

    print("HPSNR Ordered Dithering:", hpsnr(img, ordered_img))
    print("HPSNR Error Diffusion:", hpsnr(img, error_diff_img))
