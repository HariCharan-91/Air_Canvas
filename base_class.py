import cv2 as cv
from pathlib import Path

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

    # Add source_name and frame_idx so it knows HOW to save
    def __call__(self, img, source_name="webcam"):
        for step_idx , t in enumerate(self.transforms):
            # Apply the transform
            img = t(img)
            # Save the intermediate output if requested
            if self.save_output:
                # Create: current_dir / images / filename_folder /
                folder_path = Path.cwd() / 'images' / str(source_name)
                folder_path.mkdir(parents=True, exist_ok=True)
                
                # Name the file: prefix_0001.jpg
                # zfill(4) makes sure it saves as 0000, 0001, 0002 for clean sorting!
                file_name = f"{t.save_prefix}_{str(step_idx).zfill(4)}.jpg"
                full_path = folder_path / file_name
                cv.imwrite(str(full_path), img)     
        return img
    
    def __iter__(self):
        return iter(self.transforms)