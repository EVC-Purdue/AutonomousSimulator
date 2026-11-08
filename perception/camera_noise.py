"""Camera signal model on a clean render: exposure, motion blur, shot and read noise, MJPEG, then camera_node's resize."""
import cv2
import numpy as np


def apply(bgr, level, seed, yaw_rate=0.0, width=360):
    """Render (H, W, 3) uint8 -> the image camera_node would publish, seeded; level is a noise.json entry."""
    rng = np.random.default_rng(seed)
    img = bgr
    if level.get("full_well") or level.get("exposure_sigma_ev") or level.get("blur_px_per_radps"):
        lin = (img.astype(np.float32) / 255.0) ** 2.2
        lin *= 2.0 ** float(rng.normal(0.0, level.get("exposure_sigma_ev", 0.0)))
        blur = int(round(level.get("blur_px_per_radps", 0.0) * abs(yaw_rate)))
        if blur > 1:   # yaw sweeps the scene sideways during the exposure
            k = np.full((1, blur), 1.0 / blur, np.float32)
            lin = cv2.filter2D(lin, -1, k, borderType=cv2.BORDER_REFLECT)
        fw = level.get("full_well", 0)
        if fw:
            lin = np.clip(lin, 0.0, None)
            lin = lin + np.sqrt(lin / fw) * rng.standard_normal(lin.shape).astype(np.float32)
        img = np.clip(lin, 0.0, 1.0) ** (1 / 2.2) * 255.0
        img = img + level.get("read_sigma", 0.0) * rng.standard_normal(img.shape).astype(np.float32)
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    q = level.get("jpeg_quality", 0)
    if q:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(h * (width / w))))   # camera_node: 360 px wide, default interpolation
