import cv2
import numpy as np
from PIL import Image


class ImageEnhancer:
    def __call__(self, image):
        """
        Input  : PIL.Image (RGB)
        Output : PIL.Image (RGB)
        """

        # -------------------------------
        # SAFETY: FORCE PIL RGB INPUT
        # -------------------------------
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise TypeError(f"Enhancer expected PIL.Image, got {type(image)}")

        image = image.convert("RGB")

        # -------------------------------
        # PIL -> OpenCV
        # -------------------------------
        img = np.array(image)  # (H, W, 3)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Noise removal
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # -------------------------------
        # CRITICAL: BACK TO RGB
        # -------------------------------
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        # Return PIL Image (RGB)
        return Image.fromarray(enhanced_rgb)
