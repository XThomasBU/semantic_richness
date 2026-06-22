"""Rotate images for linear probe (cv2 if available, else PIL)."""

from __future__ import annotations

from pathlib import Path


def rotate_image(
    image_path: str,
    angle: float,
    output_path: str | None = None,
    image_size: int = 336,
) -> str:
    """
    Rotate image by angle (degrees), output fixed image_size x image_size canvas.
    Matches experiments.identity.spatial_illusion.rotate_image when cv2 is available.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image

        img_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_cv is None:
            img_pil = Image.open(image_path).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        height, width = img_cv.shape[0], img_cv.shape[1]
        if (width, height) != (image_size, image_size):
            img_cv = cv2.resize(img_cv, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            height, width = image_size, image_size

        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated_cv = cv2.warpAffine(
            img_cv,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        if output_path is None:
            import tempfile

            output_path = str(
                Path(tempfile.gettempdir()) / f"rotated_{Path(image_path).stem}.png"
            )
        cv2.imwrite(output_path, rotated_cv)
        return output_path
    except ImportError:
        pass

    import random
    import tempfile
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    if float(angle) != 0.0:
        img = img.rotate(-float(angle), resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    if output_path is None:
        output_path = str(
            Path(tempfile.gettempdir()) / f"rotated_{random.randint(0, 999999)}.png"
        )
    img.save(output_path, "PNG")
    return output_path
