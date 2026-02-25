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