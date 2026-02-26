import cv2 as cv
from pathlib import Path
import sys
import base_class as bc
from functools import partial
from custom import Filters , StackImages 
from media import MediaLoader


def main():
    img = cv.imread("../car.jpg")
    window = bc.TrackBar()
    window.init_trackbars()
    while True:
        mask = window.get_mask(img)
        imgres = cv.bitwise_and(img , img , mask=mask)
        cv.imshow("mask_res" , imgres)
        cv.imshow("mask" , mask)
        cv.imshow("output" , img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()