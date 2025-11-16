import sys
import cv2 as cv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def Global_DCT(img):
    img_f = np.float32(img)
    dct = cv.dct(img_f)
    idct = cv.idct(dct)
    return dct, np.uint8(np.clip(idct, 0, 255))


def Local_DCT(img, kernel_size=8):
    h, w = img.shape
    img_f = np.float32(img)

    dct_img = np.zeros((h, w), np.float32)
    idct_img = np.zeros((h, w), np.float32)

    for i in range(0, h, kernel_size):
        for j in range(0, w, kernel_size):
            block = img_f[i:i+kernel_size, j:j+kernel_size]
            dct_block = cv.dct(block)
            idct_block = cv.idct(dct_block)
            dct_img[i:i+kernel_size, j:j+kernel_size] = dct_block
            idct_img[i:i+kernel_size, j:j+kernel_size] = idct_block

    return dct_img, np.uint8(np.clip(idct_img, 0, 255))

def frequency_Domain_filter(DCT_img, keep=50):
    filtered = np.zeros_like(DCT_img)
    filtered[:keep, :keep] = DCT_img[:keep, :keep]
    idct = cv.idct(filtered)
    return filtered, np.uint8(np.clip(idct, 0, 255))

def extract_blocks(img, block_size=4):
    h, w = img.shape
    h_cut = h - h % block_size
    w_cut = w - w % block_size
    img = img[:h_cut, :w_cut]

    blocks = []
    for i in range(0, h_cut, block_size):
        for j in range(0, w_cut, block_size):
            block = img[i:i+block_size, j:j+block_size]
            blocks.append(block.flatten())
            
    return np.array(blocks), (h_cut, w_cut)

def lbg_codebook_training(vectors, codebook_size=64, epsilon=1e-3, max_iter=100):
    np.random.seed(1)
    codebook = vectors[np.random.choice(len(vectors), 1)]

    while len(codebook) < codebook_size:
        codebook = np.vstack((codebook * 1.01, codebook * 0.99))

        for _ in range(max_iter):
            dist = np.linalg.norm(vectors[:, None] - codebook[None, :], axis=2)
            labels = np.argmin(dist, axis=1)

            new_codebook = []
            for i in range(len(codebook)):
                cluster = vectors[labels == i]
                if len(cluster) > 0:
                    new_codebook.append(np.mean(cluster, axis=0))
                else:
                    new_codebook.append(codebook[i])
            new_codebook = np.array(new_codebook)

            if np.linalg.norm(new_codebook - codebook) < epsilon:
                break
            codebook = new_codebook

    return codebook


def vq_encode(vectors, codebook):
    dist = np.linalg.norm(vectors[:, None] - codebook[None, :], axis=2)
    return np.argmin(dist, axis=1)


def vq_decode(indices, codebook, image_shape, block_size=4):
    h, w = image_shape
    rec_img = np.zeros((h, w))
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            rec_img[i:i+block_size, j:j+block_size] = codebook[indices[idx]].reshape(block_size, block_size)
            idx += 1
    return np.uint8(np.clip(rec_img, 0, 255))


def visualize_codebook_as_table(codebook):
    df = pd.DataFrame(codebook.astype(int))
    df.to_csv("codebook.csv", index=False)

if __name__ == '__main__':
    img_path = 'F-16'
    img = cv.imread('images/' + img_path + '-image.png', cv.IMREAD_GRAYSCALE)

    # ========== DCT ========== #
    dct_global, idct_global = Global_DCT(img)
    filtered_DCT, idct_filtered = frequency_Domain_filter(dct_global, keep=50)

    cv.imwrite('result/'+ img_path +'_idct_global.png', idct_global)
    cv.imwrite('result/'+ img_path +'_idct_filtered.png', idct_filtered)

    # ========== VQ ========== #
    vectors, shape = extract_blocks(img, block_size=4)
    codebook = lbg_codebook_training(vectors, codebook_size=64)
    indices = vq_encode(vectors, codebook)
    rec_img = vq_decode(indices, codebook, shape, block_size=4)

    cv.imwrite('result/'+ img_path +"_VQ_reconstructed.png", rec_img)
    visualize_codebook_as_table(codebook)
