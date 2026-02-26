import cv2 as cv
from pathlib import Path
import sys
import base_class as bc
from functools import partial
from custom import Filters as fl
from custom import StackImages
import numpy as np

class MediaLoader:
    def __init__(self) -> None:
        pass

    def imgloader(self, path: str) -> np.ndarray:
        if not path:
            raise ValueError("You must provide a valid file path.")
            
        self.path_checker(path)
        img = cv.imread(path)

        if img is None:
            raise FileNotFoundError(f"Error: OpenCV could not read '{path}'.")
        return img
    
    def videoloader(self , choice , width = 640, height = 480):

        if isinstance( choice , str):
            self.path_checker(choice)
        
        vid = cv.VideoCapture(choice)
        vid.set(cv.CAP_PROP_FRAME_WIDTH, width)
        vid.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        if not vid.isOpened():
            sys.exit("Error: Could not open video source.")
        
        return vid
    
    def mask_finder(self, window : str = "Track"):
        trackbar = bc.TrackBar( window_name = window)
        trackbar.init_trackbars()
        return trackbar

    def path_checker(self , choice) -> None:
        """Checks if the provided path is a valid file."""
        if isinstance(choice, str):
            path_obj = Path(choice)
            if not path_obj.is_file():
                # Raising an error 
                raise FileNotFoundError(f"Error: File '{choice}' not found.")
            

    def video_transform(self , vid = None , flip = None , pipeline = None , mask = None) -> None:
        """Loads and displays a video or webcam feed."""
        try:
            while True:
                is_true, img = vid.read()
                
                if not is_true:
                    print("Video ended or source lost.")
                    break

                # Check against None so that 0 (vertical flip) is still processed
                if flip is not None:
                    img = cv.flip(img, flip)

                if pipeline:
                    img = pipeline(img, source_name="video")
                
                cv.imshow("Video Feed", img)

                # Wait for 5ms, exit if 'q' is pressed
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Ensure resources are ALWAYS released, even if the loop breaks or errors
            vid.release()
            cv.destroyAllWindows()

    def image_transform(self , img : np.ndarray , flip = None , pipeline = None, stack = False) -> np.ndarray:
        """Loads and displays a single image."""
        
        if flip is not None:
            img = cv.flip(img, flip)

        if pipeline:
            # 1. Get the clean name of the file (e.g., "../car.jpg" -> "car")
            # source_name = Path(choice).stem if isinstance(choice, str) else "image"
            
            # 2. Call the pipeline directly! DO NOT loop over it.
            # This triggers Compose.__call__ and runs your save logic.
            img = pipeline(img , source_name = "image")

        return img
        
        # if stack:
        #     path = rf"C:\Users\lenovo\projects\cv\air_canvas\images\{source_name}"
        #     grid = StackImages(path).stack()
        #     if grid is not None and isinstance(grid, np.ndarray):
        #         cv.imshow(f"Pipeline Grid", grid)
        #         cv.waitKey(0)
        #         cv.destroyAllWindows()
        #     else:
        #         print("Error: Grid generation failed.")
        # else:
        #     cv.imshow(f"final_image", img)
        #     cv.waitKey(0)
        #     cv.destroyAllWindows()

    def find_mask_image(self, img: np.ndarray, trackbar, pipeline=None):
        """Loops a single image with trackbars to find the perfect mask."""
        print("Press 'q' to exit and save mask values.")
        while True:
            # Apply pipeline if you have one (like your HSV filter), 
            # otherwise just use the raw image
            processed_img = pipeline(img.copy(), source_name="image") if pipeline else img
            
            mask, values = trackbar.get_mask(processed_img)
            result = cv.bitwise_and(img, img, mask=mask)

            cv.imshow("Original", img)
            cv.imshow("Mask", mask)
            cv.imshow("Masked Result", result)

            if cv.waitKey(1) & 0xFF == ord('q'):
                print(f"Final Mask Values: {values}")
                break
                
        cv.destroyAllWindows()
        return values

    def find_mask_video(self, vid, trackbar, flip=None, pipeline=None):
        """Plays video feed with trackbars to find the perfect mask."""
        print("Press 'q' to exit and save mask values.")
        values = None
        try:
            while True:
                is_true, img = vid.read()
                if not is_true:
                    print("Video ended or source lost.")
                    break

                if flip is not None:
                    img = cv.flip(img, flip)

                processed_img = pipeline(img.copy(), source_name="video") if pipeline else img

                mask, values = trackbar.get_mask(processed_img)
                result = cv.bitwise_and(img, img, mask=mask)

                # cv.imshow("Video Feed", img)
                # cv.imshow("Mask", mask)
                cv.imshow("Masked Result", result)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    print(f"Final Mask Values: {values}")
                    break
        finally:
            vid.release()
            cv.destroyAllWindows()
            
        return values

def main() -> None:
    media_load = MediaLoader()
    
    # --- For Video ---
    vid = media_load.videoloader(0 , width=640 , height=480)
    trackbar = media_load.mask_finder("Video Trackbars")
    
    pipeline = bc.Compose([
        fl.hsv()
    ], save_output=True)

    try:
        # Just call this ONE line to do all the video masking!
        # final_values = media_load.find_mask_video(vid, trackbar, pipeline=pipeline, flip=1)
        # print("Values to use in your code:", final_values)
        color_dict ={
            "Blue" : ([109, 117, 213], [179, 255, 255]),
            "Red" : ([154, 90, 129], [179, 255, 255])
        }

        contour = cp.Co
        
    except Exception as e:
        print(f"An error occurred: {e}")

    # --- For Image (Whenever you need it) ---
    # img = media_load.imgloader("../car.jpg")
    # img_trackbar = media_load.mask_finder("Image Trackbars")
    # final_img_values = media_load.find_mask_image(img, img_trackbar, pipeline=pipeline)

if __name__ == "__main__":
    main()