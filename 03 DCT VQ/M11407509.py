import sys
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Option 1: DCT
# =========================

def Global_DCT(img):
    # TODO:
    #   1. Use OpenCV’s DCT function or numpy’s fft function
    #      or implement according to the given DCT formula.
    #      e.g., cv.dct(img_float32) <--> cv.idct(dct_img)
    #   2. Return both DCT result and reconstructed image (optional)
    pass


def Local_DCT(img, kernel_size=8):
    # TODO:
    #   1. Divide the image into non-overlapping blocks (kernel_size x kernel_size)
    #   2. Apply 2D DCT to each block (using OpenCV / numpy / formula)
    #   3. Reconstruct the image from block-wise IDCT (optional)
    pass


def frequency_Domain_filter(DCT_img):
    # TODO:
    #   1. Input the DCT frequency-domain image
    #   2. Crop the low-frequency area (upper-left corner, size is arbitrary)
    #   3. Set other coefficients to zero (or keep as is)
    #   4. Apply IDCT to reconstruct the filtered image
    #   5. Return the filtered DCT and reconstructed image
    pass


# =========================
# Option 2: Vector Quantization (VQ)
# =========================

def extract_blocks(img, block_size=4):
    # TODO:
    #   1. Divide the image into non-overlapping blocks of size (block_size x block_size)
    #   2. Reshape each block into a vector and store all vectors in an array
    #   3. Return the array of vectors and possibly the image shape for reconstruction
    pass


def lbg_codebook_training(vectors, codebook_size=64, epsilon=1e-3, max_iter=100):
    # TODO (Linde–Buzo–Gray algorithm, as in Chapter 7, Slide 46):
    #   1. Initialize the codebook (e.g., random selection of vectors)
    #   2. Repeat until convergence:
    #       a. Assign each vector to the nearest codeword (cluster assignment)
    #       b. Update each codeword as the mean of its assigned vectors
    #       c. Compute distortion and check stopping condition
    #   3. Return the final codebook
    pass


def vq_encode(vectors, codebook):
    # TODO:
    #   1. For each vector, find the nearest codeword in the codebook
    #   2. Store the index of the nearest codeword
    #   3. Return the index array (VQ encoded representation)
    pass


def vq_decode(indices, codebook, image_shape, block_size=4):
    # TODO:
    #   1. Map each index back to its corresponding codeword (vector)
    #   2. Reshape each codeword vector back to (block_size x block_size)
    #   3. Reconstruct the full image from all blocks using image_shape
    #   4. Return the reconstructed image
    pass


def visualize_codebook_as_table(codebook, block_size=4):
    # TODO:
    #   1. Optionally print or save the codebook as a table (for report screenshot)
    #   2. For example, each row = one codeword (flattened vector)
    #   3. Or format as a grid and save as an image
    pass


if __name__ == '__main__':

    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    # =========================
    # Option 1: DCT (Global / Local / Frequency Filtering)
    # =========================
    # TODO:
    #   1. Call Global_DCT / Local_DCT
    #   2. Call frequency_Domain_filter on the DCT result
    #   3. Show and/or save:
    #       - Original image
    #       - DCT image (frequency domain)
    #       - IDCT reconstructed image
    #       - Frequency-domain filtered + IDCT image

    # =========================
    # Option 2: VQ (LBG-based Vector Quantization)
    # =========================
    # TODO:
    #   1. Extract blocks and convert to vectors: extract_blocks()
    #   2. Train codebook using LBG: lbg_codebook_training()
    #   3. Encode the image: vq_encode()
    #   4. Decode and reconstruct the image: vq_decode()
    #   5. Visualize the codebook as a table for the report: visualize_codebook_as_table()
    #   6. Show and/or save:
    #       - Original image
    #       - VQ reconstructed image
    #       - Codebook table screenshot (or exported figure)

    pass