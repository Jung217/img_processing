import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def Global_HE(img):
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    pdf = hist / np.sum(hist)
    cdf = pdf.cumsum()
    equalized_img = (cdf[img] * 255).astype(np.uint8)
    
    return equalized_img, hist, np.histogram(equalized_img.flatten(), 256, [0,256])[0]


def Local_HE(img, window_size=31):
    pad = window_size // 2
    padded = np.pad(img, pad, mode='reflect')
    dst = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+window_size, j:j+window_size]
            hist, _ = np.histogram(window.flatten(), 256, [0,256])
            pdf = hist / np.sum(hist)
            cdf = pdf.cumsum()
            dst[i, j] = np.clip(cdf[img[i,j]] * 255, 0, 255)
    
    return dst.astype(np.uint8)


def plot_hist(before, after, title, filename):
    hist_before, _ = np.histogram(before.flatten(), 256, [0,256])
    hist_after, _ = np.histogram(after.flatten(), 256, [0,256])

    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.bar(np.arange(256), hist_before, alpha=0.6, label="Before")
    plt.bar(np.arange(256), hist_after, alpha=0.6, label="After")

    plt.legend()
    
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved histogram image: {filename}")



if __name__ == '__main__':
    img_path = 'F-16'
    img = cv.imread('images/' + img_path + '-image.png', cv.IMREAD_GRAYSCALE)

    global_img, hist_b, hist_a = Global_HE(img)
    local_img = Local_HE(img)

    # cv.imshow("Original", img)
    # cv.imshow("Global HE", global_img)
    # cv.imshow("Local HE", local_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    cv.imwrite('result/'+ img_path +'_global.png', global_img)
    cv.imwrite('result/'+ img_path +'_local.png', local_img)

    plot_hist(img, global_img, "Histogram Before vs After Global HE",'result/'+ img_path +'_HE_global.png')
    plot_hist(img, local_img, "Histogram Before vs After Local HE",'result/'+ img_path +'_HE_local.png')