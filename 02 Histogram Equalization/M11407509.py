import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def Global_HE(img):
    #TODO:
        # step 1 : Count the number of pixel occurrences (hint : numpy --> unique())
        # step 2 : Calculate the histogram equalization
        # step 3 : Display histogram(comparison before and after equalization)
    pass


def Local_HE(img):
    # TODO:
        # step 1 : Count the number of pixel occurrences
        # step 2 : Define a square neighborhood and move the center of this area from pixel to pixel.
        # step 3 : Calculate the histogram equalization
    pass
    
 
if __name__ == '__main__':

    img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    # TODO: Display histogram(comparison before and after equalization)
    