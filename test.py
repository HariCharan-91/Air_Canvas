import cv2 as cv
from pathlib import Path
import sys
import base_class as bc
from functools import partial
from custom import Filters , StackImages
from media import MediaLoader


def main():
    path = Path(r"C:\Users\lenovo\projects\cv\air_canvas\images\car")

    if not path.exists():
        print("path not found")

    mat = StackImages(folder_path = path)
    mat.stack()

if __name__ == "__main__":
    main()