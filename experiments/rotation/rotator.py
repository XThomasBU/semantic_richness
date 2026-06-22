import os
import cv2
import numpy as np
from scipy.ndimage import rotate

class ExactRotator:
    def __init__(self, input_folder, output_folder, bg_value=255):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.dataset_name = os.path.basename(os.path.normpath(input_folder))
        self.bg_value = bg_value

    def rotate_image_exact(self, image, angle):
        return rotate(
            image, 
            angle, 
            order=0,           # Nearest neighbor - no interpolation
            reshape=True,      # Expand canvas to fit rotated image
            cval=self.bg_value, # Fill value for new pixels
            prefilter=False    # Disable any preprocessing
        ).astype(image.dtype)

    def rotate_image_smooth(self, image, angle, resampling=cv2.INTER_LINEAR, ref_box=False, reshape=False):
        height, width = image.shape[0], image.shape[1]
        center = (width // 2, height // 2)

        if ref_box:
            # Add a reference box to the image just inside the border of the image
            cv2.rectangle(image, (1, 1), (width - 2, height - 2), (0,), thickness=1)

        # Rotate image using opencv
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        if reshape:
            # Compute new bounding dimensions
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            new_width = int(height * abs_sin + width * abs_cos)
            new_height = int(height * abs_cos + width * abs_sin)

            # Adjust rotation matrix to take into account translation
            rotation_matrix[0, 2] += new_width / 2 - center[0]
            rotation_matrix[1, 2] += new_height / 2 - center[1]

            width, height = new_width, new_height

        rotated_image = cv2.warpAffine(image, 
                                       rotation_matrix, 
                                       (width, height), 
                                       flags=resampling,
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=(255,255,255))

        # Round greyscale values to 0 or 255
        flattened_image = rotated_image.flatten()
        for idx, val in enumerate(flattened_image):
            if val != 0 and val != 255:
                val = round(val / 255.0) * 255
                flattened_image[idx] = val

        return flattened_image.reshape(rotated_image.shape).astype(image.dtype)

    def _parse_path(self, rel_path):
        parts = rel_path.split(os.sep)
        if len(parts) < 3:
            return None
        if len(parts) == 3:
            script_name, character, image_name = parts
            stem, _ = os.path.splitext(image_name)
            character_idx = stem
        else:
            script_name, character, character_idx, image_name = parts[-4:]
        return script_name, character, character_idx, image_name

    def process(self, angles=range(10, 100, 10)):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        
        for root, _, files in os.walk(self.input_folder):
            for file in files:
                if not file.lower().endswith(valid_exts):
                    continue
                    
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, self.input_folder)
                parsed = self._parse_path(rel_path)
                
                if parsed is None:
                    print(f"Skipping (unexpected path): {img_path}")
                    continue

                # Read as grayscale
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Skipping (unreadable): {img_path}")
                    continue

                script_name, character, character_idx, image_name = parsed
                out_name = image_name


                for angle in angles:
                    # Exact rotation with no interpolation
                    # rotated = self.rotate_image_exact(image, angle)

                    # Smooth rotation with interpolation
                    rotated = self.rotate_image_smooth(image, angle, resampling=cv2.INTER_LINEAR, ref_box=False, reshape=False)
                    
                    save_path = os.path.join(
                        self.output_folder,
                        script_name,
                        character,
                        character_idx,
                        f"Angle_{angle}",
                        out_name
                    )
                    
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    ok = cv2.imwrite(save_path, rotated)
                    
                    print(("Saved" if ok else "Failed to save"), save_path)

# Usage
if __name__ == "__main__":
    from pathlib import Path
    _repo = Path(__file__).resolve().parents[2]
    rotator = ExactRotator(
        input_folder=str(_repo / "DATA" / "omniglot-master" / "python" / "images_original"),
        output_folder=str(_repo / "DATA" / "omniglot-master" / "python" / "images_rotated"),
        bg_value=255,  # White background
    )
    rotator.process(angles=range(0, 100, 10))