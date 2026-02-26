import cv2 as cv
from pathlib import Path
import numpy as np

class Layer:
    def __init__(self, func, save_prefix="layer", **kwargs):
        self.func = func
        self.kwargs = kwargs
        # Give every layer a default prefix so it doesn't crash if you forget it
        self.save_prefix = save_prefix

    def __call__(self, img):
        return self.func(img, **self.kwargs)
    
class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms, save_output=False):
        self.transforms = transforms
        self.save_output = save_output

    def __call__(self, img, source_name="webcam"):
        # --- NEW: Cleanup Logic ---
        if self.save_output:
            folder_path = Path.cwd() / 'images' / str(source_name)
            
            # If folder exists, delete everything inside it
            if folder_path.exists():
                for file in folder_path.iterdir():
                    try:
                        if file.is_file():
                            file.unlink() # This deletes the file
                    except Exception as e:
                        print(f"Could not delete {file}: {e}")
            else:
                # Create it if it doesn't exist
                folder_path.mkdir(parents=True, exist_ok=True)

        # --- Proceed with Pipeline ---
        for step_idx, t in enumerate(self.transforms, start=1):
            img = t(img)
            
            if self.save_output:
                # Now we save fresh images into a clean folder
                file_name = f"{str(step_idx).zfill(2)}_{t.save_prefix}.jpg"
                full_path = folder_path / file_name
                cv.imwrite(str(full_path), img)
        return img
    
    def __iter__(self):
        return iter(self.transforms)

class TrackBar:
    """Class to manage OpenCV Trackbars for HSV color masking."""
    
    def __init__(self, window_name="Track", width=640, height=240):
        self.window_name = window_name
        self.width = width
        self.height = height

    @staticmethod
    def empty(a):
        pass

    def init_trackbars(self):
        """Creates the window and the trackbars. Run this once."""
        cv.namedWindow(self.window_name)
        cv.resizeWindow(self.window_name, self.width, self.height)
        
        # cv.createTrackbar(trackbar_name, window_name, default_value, max_value, callback)
        cv.createTrackbar("hue min", self.window_name, 0, 179, self.empty)
        cv.createTrackbar("hue max", self.window_name, 179, 179, self.empty)
        cv.createTrackbar("sat min", self.window_name, 0, 255, self.empty)
        cv.createTrackbar("sat max", self.window_name, 255, 255, self.empty)
        cv.createTrackbar("value min", self.window_name, 0, 255, self.empty)
        cv.createTrackbar("value max", self.window_name, 255, 255, self.empty)
        

    def get_mask(self, img: np.ndarray) -> np.ndarray:
        """Reads trackbar positions and applies the mask. Run this in your video loop."""
        
        # Read the current positions directly using the string names
        h_min = cv.getTrackbarPos("hue min", self.window_name)
        h_max = cv.getTrackbarPos("hue max", self.window_name)
        s_min = cv.getTrackbarPos("sat min", self.window_name)
        s_max = cv.getTrackbarPos("sat max", self.window_name)
        v_min = cv.getTrackbarPos("value min", self.window_name)
        v_max = cv.getTrackbarPos("value max", self.window_name)
        
        lower = np.array([h_min, s_min, v_min])
        higher = np.array([h_max, s_max, v_max])
        
        mask = cv.inRange(img, lower, higher)
        return mask , [[h_min , s_min ,v_min] ,[h_max , s_max ,v_max]]
    