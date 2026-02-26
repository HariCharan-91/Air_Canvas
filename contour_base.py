import cv2 as cv
import numpy as np 

class ContourHandler:
    def __init__(self , color_dict : dict , min_area : int = 500) -> None:
        """
        Initializes the handler with a dictionary of colors to detect.
        Format: {"Color Name": ([h_min, s_min, v_min], [h_max, s_max, v_max])}
        """
        self.color_dict = color_dict
        self.min_area = min_area

    def process(self , img : np.ndarray , img_hsv : np.ndarray):
        contour_image = img.copy()

        for color_name, (lower_vals, upper_vals) in self.color_dict.items():
            lower_bound = np.array(lower_vals)
            upper_bound = np.array(upper_vals)

            # 1. Create mask for this color
            mask = cv.inRange(img_hsv, lower_bound, upper_bound)

            # 2. Find contours
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # 3. Draw bounding boxes
            for cnt in contours:
                if cv.contourArea(cnt) > self.min_area:
                    x, y, w, h = cv.boundingRect(cnt)
                    
                    cv.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(contour_image, color_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
            return contour_image