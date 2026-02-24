import cv2 as cv
import numpy as np
import base_class as bc

class Filters:
    """Library of custom OpenCV pipeline layers."""
    
    @staticmethod
    def gray() -> bc.Layer:
        return bc.Layer(cv.cvtColor, code=cv.COLOR_BGR2GRAY, save_prefix="gray")

    @staticmethod
    def hsv() -> bc.Layer:
        return bc.Layer(cv.cvtColor, code=cv.COLOR_BGR2HSV, save_prefix="hsv")

    @staticmethod
    def blur(ksize: tuple = (7, 7), sigmaX: int = 3) -> bc.Layer:
        """Applies Gaussian Blur."""
        return bc.Layer(cv.GaussianBlur, save_prefix="blur", ksize=ksize, sigmaX=sigmaX)

    @staticmethod
    def edges(t1: int = 100, t2: int = 200) -> bc.Layer:
        """Applies Canny Edge Detection."""
        return bc.Layer(cv.Canny, save_prefix="edges", threshold1=t1, threshold2=t2)

    @staticmethod
    def resize(dsize: tuple = (0, 0), fx: float = 0.5, fy: float = 0.5, interp=cv.INTER_LINEAR) -> bc.Layer:
        """Resizes the image."""
        return bc.Layer(cv.resize, dsize=dsize, fx=fx, fy=fy, interpolation=interp, save_prefix="resized")

    @staticmethod
    def erode(kernel: np.ndarray = None, iterations: int = 1) -> bc.Layer:
        """
        Applies Erosion (shrinks white regions).
        Defaults to a 3x3 square kernel.
        """
        if kernel is None:
            kernel = np.ones((3, 3), np.uint8)
        return bc.Layer(cv.erode, save_prefix="eroded", kernel=kernel, iterations=iterations)

    @staticmethod
    def dilate(kernel: np.ndarray = None, iterations: int = 1) -> bc.Layer:
        """
        Applies Dilation (expands white regions).
        Defaults to a 3x3 square kernel.
        """
        if kernel is None:
            kernel = np.ones((3, 3), np.uint8)
        return bc.Layer(cv.dilate, save_prefix="dilated", kernel=kernel, iterations=iterations)