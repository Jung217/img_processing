import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def generate_bayer_matrix(n):
    if n == 1: return np.array([[0, 2], [3, 1]])
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
            if j + 1 < w: img[i, j+1] += quant_error * 7/16
            if i + 1 < h:
                if j > 0: img[i+1, j-1] += quant_error * 3/16
                img[i+1, j] += quant_error * 5/16
                if j + 1 < w: img[i+1, j+1] += quant_error * 1/16
    return Error_Diffusion_img.astype(np.uint8)

def Dot_Diffusion(img):
    img = img.astype(float)
    h, w = img.shape

    class_matrix = np.array([
        [26,15,11,42,41,37,12, 7],
        [27,32,33, 5,10,16,23, 4],
        [13,17,18,19,20,21,14,20],
        [ 6,45,44, 8, 0,29,39,47],
        [49,34,25,43,38,28,24,48],
        [51, 9,36,35,31,22, 1,50],
        [53,46, 3, 2,30,56,58,52],
        [55,57,59,60,62,63,61,54]
    ])
    tile_h = (h + 7) // 8
    tile_w = (w + 7) // 8
    class_matrix = np.tile(class_matrix, (tile_h, tile_w))[:h, :w]

    N = class_matrix.max() + 1
    halftone = np.zeros_like(img)
    processed = np.zeros_like(img, dtype=bool)

    for t in range(N):
        mask = (class_matrix == t)
        coords = np.argwhere(mask)

        for (i, j) in coords:
            new_pixel = 255 if img[i, j] > 127 else 0
            halftone[i, j] = new_pixel
            processed[i, j] = True
            error = img[i, j] - new_pixel
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and not processed[ni, nj]:
                        img[ni, nj] += error / 8

    return halftone.astype(np.uint8)

# def DBS(img, iterations=2, kernel_size=7):
#     img = img.astype(float)
#     h, w = img.shape
#     halftone = (img > 127).astype(float) * 255

#     kernel = cv.getGaussianKernel(kernel_size, kernel_size / 3)
#     kernel = kernel @ kernel.T

#     for _ in range(iterations):
#         for i in range(h):
#             for j in range(w):
#                 current = halftone[i, j]
#                 halftone[i, j] = 255 - current

#                 filtered = cv.filter2D(halftone, -1, kernel)
#                 mse_new = np.mean((filtered - img) ** 2)
#                 halftone[i, j] = current
#                 filtered_old = cv.filter2D(halftone, -1, kernel)
#                 mse_old = np.mean((filtered_old - img) ** 2)

#                 if mse_new < mse_old: halftone[i, j] = 255 - current
#     return halftone.astype(np.uint8)

def hpsnr(original, halftoned):
    mse = np.mean((original.astype(float) - halftoned.astype(float)) ** 2)
    if mse == 0: return float('inf')
    return round(10 * np.log10(255**2 / mse), 5)

if __name__ == '__main__':
    img_path = 'Baboon'
    img = cv.imread('images/' + img_path + '-image.png', cv.IMREAD_GRAYSCALE)
    n = 2
    bayer_matrix = generate_bayer_matrix(n)
    thresholds_matrix = generate_thresholds_matrix(bayer_matrix)

    ordered_img = Ordered_Dithering(img, thresholds_matrix)
    error_diff_img = Error_Diffusion(img)
    #dbs_img = DBS(img)
    dotdiff_img = Dot_Diffusion(img)

    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1); plt.title('Original'); plt.imshow(img, cmap='gray'); plt.axis('off')
    plt.subplot(2,2,2); plt.title('Ordered Dithering'); plt.imshow(ordered_img, cmap='gray'); plt.axis('off')
    plt.subplot(2,2,3); plt.title('Error Diffusion'); plt.imshow(error_diff_img, cmap='gray'); plt.axis('off')
    plt.subplot(2,2,4); plt.title('Dot Diffusion'); plt.imshow(dotdiff_img, cmap='gray'); plt.axis('off')
    plt.show()

    cv.imwrite('result/'+ img_path +'_dithering.png', ordered_img)
    cv.imwrite('result/'+ img_path +'_diffusion.png', error_diff_img)
    #cv.imwrite('result/F16_DBS.png', dbs_img)
    cv.imwrite('result/'+ img_path +'_DotDiffusion.png', dotdiff_img)

    print(img_path)
    print("HPSNR Ordered Dithering:", hpsnr(img, ordered_img))
    print("HPSNR Error Diffusion:", hpsnr(img, error_diff_img))
    #print("HPSNR DBS:", hpsnr(img, dbs_img))
    print("HPSNR Dot Diffusion:", hpsnr(img, dotdiff_img))