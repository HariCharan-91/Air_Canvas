import cv2 as cv
from pathlib import Path
import sys
import base_class as bc
from functools import partial
from custom import Filters as fl

class MediaLoader:
    def __init__(self,width=640, height=480) -> None:
        self.width = width
        self.height = height

    def path_checker(self , choice) -> None:
        """Checks if the provided path is a valid file."""
        if isinstance(choice, str):
            path_obj = Path(choice)
            if not path_obj.is_file():
                # Raising an error 
                raise FileNotFoundError(f"Error: File '{choice}' not found.")

    def video_loader(self , choice = 0 , flip = None , pipeline = None) -> None:
        """Loads and displays a video or webcam feed."""
        self.path_checker(choice)

        vid = cv.VideoCapture(choice)
        
        vid.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        vid.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

        if not vid.isOpened():
            sys.exit("Error: Could not open video source.")

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
                    source_name = Path(choice).stem if isinstance(choice, str) else "webcam"
                    img = pipeline(img, source_name=source_name)

                cv.imshow("Video Feed", img)

                # Wait for 5ms, exit if 'q' is pressed
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Ensure resources are ALWAYS released, even if the loop breaks or errors
            vid.release()
            cv.destroyAllWindows()

    def image_loader(self , choice , flip = None , pipeline = None) -> None:
        """Loads and displays a single image."""
        self.path_checker(choice)
        
        img = cv.imread(choice)
        
        if img is None:
            sys.exit(f"Error: Could not read image at {choice}")

        if flip is not None:
            img = cv.flip(img, flip)

        if pipeline:
            # 1. Get the clean name of the file (e.g., "../car.jpg" -> "car")
            source_name = Path(choice).stem if isinstance(choice, str) else "image"
            
            # 2. Call the pipeline directly! DO NOT loop over it.
            # This triggers Compose.__call__ and runs your save logic.
            img = pipeline(img , source_name = source_name)
                    
        cv.imshow("Image", img)
        cv.waitKey(0)
        cv.destroyAllWindows() # Clean up window after key press

def main() -> None:
    media_load = MediaLoader()
    pipeline = bc.Compose([
        fl.hsv()
    ] )
    try:
        media_load.video_loader(choice = 0, pipeline = pipeline , flip=1)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()