import cv2 as cv
import numpy as np
import base_class as bc
from pathlib import Path
import re
import tkinter
from pathlib import Path
from functools import wraps


class Filters:
    """Library of custom OpenCV pipeline layers."""

    @staticmethod
    def view_copy() -> bc.Layer:
        # Safely passes a clone of the image down the pipeline
        return bc.Layer(np.copy, save_prefix="unmodified_copy")
    
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
    
    @staticmethod
    def hsv_mask(lower_list: list, upper_list: list):
        # This inner function is what actually touches the pixels
        def apply_logic(img):
            img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            lower = np.array(lower_list)
            upper = np.array(upper_list)
            mask = cv.inRange(img_hsv, lower, upper)
            # We return the bitwise result (the actual masked image)
            return cv.bitwise_and(img, img, mask=mask)
        return bc.Layer(apply_logic, save_prefix="hsv_mask")

def get_screen_limits(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        root = tkinter.Tk()
        kwargs['max_w'] = root.winfo_screenwidth()
        kwargs['max_h'] = root.winfo_screenheight()
        root.destroy()
        return func(self, *args, **kwargs)
    return wrapper

class StackImages:
    def __init__(self, folder_path) -> None:
        self.folder_path = Path(folder_path)

    @get_screen_limits
    def stack(self, cols=3, **kwargs):
        max_w, max_h = kwargs.get('max_w'), kwargs.get('max_h')

        # DEBUG 1: Is the folder path even correct?
        if not self.folder_path.exists():
            print(f"❌ ERROR: Folder does not exist: {self.folder_path}")
            return

        def get_num(p):
            match = re.search(r'\d+', p.name)
            return int(match.group()) if match else 0
        
        # DEBUG 2: Are there any files here?
        all_files = list(self.folder_path.iterdir())
        print(f"📂 Found {len(all_files)} total files in folder.")

        paths = sorted([p for p in all_files if p.suffix.lower() in ['.jpg', '.png']], key=get_num)
        print(f"🖼️ Found {len(paths)} valid image files (jpg/png).")

        imgs = []
        for p in paths:
            img = cv.imread(str(p))
            if img is not None:
                imgs.append(img)
            else:
                print(f"⚠️ Failed to read: {p.name}")

        if not imgs:
            print("❌ Grid generation failed: No valid images to process.")
            return

        # Resize and combine (Same logic as before)
        h, w = imgs[0].shape[:2]
        resized_imgs = [cv.resize(img, (w, h)) for img in imgs]

        rows = []
        for i in range(0, len(resized_imgs), cols):
            row_chunk = resized_imgs[i : i + cols]
            while len(row_chunk) < cols:
                row_chunk.append(np.zeros((h, w, 3), dtype=np.uint8))
            rows.append(cv.hconcat(row_chunk))

        grid = cv.vconcat(rows)
        
        # Scaling logic
        gh, gw = grid.shape[:2]
        scale = min((max_w * 0.9) / gw, (max_h * 0.9) / gh, 1.0)
        if scale < 1.0:
            grid = cv.resize(grid, (0,0), fx=scale, fy=scale)

        print("✅ Grid successfully generated!")
        return grid
    