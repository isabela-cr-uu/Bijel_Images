import cv2
import numpy as np
from typing import Tuple, Optional

import skimage

# Image processing pipeline for Bijels Area Estimation using OpenCV
# Each step is documented and can be toggled/configured via parameters.

class ProcessingParams:
    def __init__(
        self,
        make_gray: str = "intensity",
        use_hist_eq: bool = True,
        use_gaussian: bool = True,
        gaussian_kernel: Tuple[int, int] = (5, 5),
        sigma_x: float = 0.0,
        sigma_y: float = 0.0,
        # quant_threshold: int = 128,
        threshold_info: dict = {"quant_threshold": 128},
        threshold_algorithm: str = "manual",
        auto_update: bool = True,
        overlay_use_zero: bool = False,
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
    ):
        self.make_gray = make_gray
        self.use_hist_eq = use_hist_eq
        self.use_gaussian = use_gaussian
        self.gaussian_kernel = gaussian_kernel
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.threshold_algorithm = threshold_algorithm
        self.threshold_info = threshold_info
        self.auto_update = auto_update
        # When True, overlay pixels where mask==0; otherwise where mask==255
        self.overlay_use_zero = overlay_use_zero
        # Overlay color in BGR
        self.overlay_color = overlay_color


def compute_histogram(gray: np.ndarray) -> np.ndarray:
    """
    Compute histogram for a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        MxN array of the grayscale image. Datatype of the elements in the array
        is unsigned 8 bit integer (numpy.ubyte).

    Returns
    -------
    np.ndarray
        Array of length 256, containing the histogram of the input image

    """

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten()


def to_gray(bgr: np.ndarray, make_gray: str) -> np.ndarray:
    """
    Convert BGR color image to grayscale.

    Parameters
    ----------
    bgr : np.ndarray
        Color image (MxNx3), in bgr order.
    make_gray : {"red", "green", "blue", "gray"}
        How to convert to grayscale. "red", "green", and "blue" return the
        corresponding channel of the color image. Any other input returns a 
        grayscale image computed through openCV

    Returns
    -------
    np.ndarray
        MxN array of the grayscale image.

    """
    if make_gray == "red":
        return bgr[:,:,2].copy()
    elif make_gray == "green":
        return bgr[:,:,1].copy()
    elif make_gray == "blue":
        return bgr[:,:,0].copy()
    else:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    


def equalize_histogram(gray: np.ndarray, enabled: bool) -> np.ndarray:
    """
    Equalize the histogram of a grayscale image if enabled.

    Parameters
    ----------
    gray : np.ndarray
        MxN array of the grayscale image. Datatype of the elements in the array
        is unsigned 8 bit integer (numpy.ubyte).
    enabled : bool
        Whether to perform this operation or not.

    Returns
    -------
    np.ndarray
        MxN array of the processed grayscale image.

    """
    
    if not enabled:
        return gray
    return cv2.equalizeHist(gray)


def gaussian_smooth(gray: np.ndarray, enabled: bool, kernel: Tuple[int, int], sigma_x: float, sigma_y: float) -> np.ndarray:
    """
    Apply Gaussian denoising/smoothing if enabled.    

    Parameters
    ----------
    gray : np.ndarray
        MxN array of the grayscale image. Datatype of the elements in the array
        is unsigned 8 bit integer (numpy.ubyte).
    enabled : bool
        Whether to perform this operation or not.
    kernel : Tuple[int, int]
        Size of the kernel to be used in the Gaussian smoothing.
    sigma_x : float
        Standard deviation for Gaussian kernel in the x direction.
    sigma_y : float
        Standard deviation for Gaussian kernel in the y direction.

    Returns
    -------
    TYPE
        MxN array of the processed grayscale image.

    """
    if not enabled:
        return gray
    kx, ky = kernel
    # Ensure odd kernel sizes for GaussianBlur; if even, increment to next odd
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1
    return cv2.GaussianBlur(gray, (kx, ky), sigmaX=sigma_x, sigmaY=sigma_y)


def binarize(gray: np.ndarray, algorithm: str, info: dict) -> Tuple[np.ndarray, np.ndarray | int ]:
    """
    Threshold the image into 2 intensities

    Parameters
    ----------
    gray : np.ndarray
        MxN array of the grayscale image. Datatype of the elements in the array
        is unsigned 8 bit integer (numpy.ubyte).
    algo : {"manual", "Otsu", "local"}
        What thresholding algorithm to use.
    params : dict
        Dictionary containing at least the arguments needed for the chosen
        thresholding algorithm

    Returns
    -------
    binary : np.ndarray
        MxN array containing Bools, True wherever the intensity in
        the original image is high, False wherever it is low.
    thr : int or np.ndarray
        The value of the threshold used. If the "local" algorithm is chosen,
        this is an MxN array of thresholds instead

    """
    if algorithm == "Otsu":
        thr = skimage.filters.threshold_otsu(gray)
    elif algorithm == "local":
        block = info["local_block"]
        #ensuring that the block_size is odd
        if block%2 == 0:
            block += 1
        thr = skimage.filters.threshold_local(gray, block_size=block, 
                                              offset=info["local_offset"])
    else:
        thr = int(np.clip(info["quant_threshold"], 0, 255))
        
    binary = gray>thr
    return binary, thr

def overlay_mask_on_color(
    color_img: np.ndarray,
    mask: np.ndarray,
    overlay_color: Tuple[int, int, int] = (0, 255, 0),
    use_zero: bool = True,
) -> np.ndarray:
    """
    Create the overlay image

    Parameters
    ----------
    color_img : np.ndarray
        MxNx3 array containing the initial (color) image.
    mask : np.ndarray
        MxN array containing Bools.
    overlay_color : Tuple[int, int, int], optional
        What color to use. 0<=n<=255 in bgr order. The default is (0, 255, 0).
    use_zero : bool, optional
        Whether to invert the mask. The default is True.

    Returns
    -------
    overlay : np.ndarray
        MxNx3 array containing the color image with the overlay applied.

    """
    
    overlay = color_img.copy()
    if use_zero:
        mask = ~mask #invert boolean values in mask
    overlay[mask] = overlay_color
    return overlay


def process_image(
    bgr: np.ndarray,
    params: ProcessingParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run processing pipeline

    Parameters
    ----------
    bgr : np.ndarray
        DESCRIPTION.
    params : ProcessingParams
        DESCRIPTION.

    Returns
    -------
    input_color : TYPE
        DESCRIPTION.
    gray : TYPE
        DESCRIPTION.
    eq_gray : TYPE
        DESCRIPTION.
    smooth_gray : TYPE
        DESCRIPTION.
    quant_mask : TYPE
        DESCRIPTION.
    final_overlay : TYPE
        DESCRIPTION.
    stats : TYPE
        DESCRIPTION.

    """
    input_color = bgr
    gray = to_gray(input_color, params.make_gray)


    # Step: Equalize Histogram (conditional)
    eq_gray = equalize_histogram(gray, params.use_hist_eq)

    # Step: Apply Gaussian denoising / smoothing (conditional)
    smooth_gray = gaussian_smooth(eq_gray, params.use_gaussian, params.gaussian_kernel, params.sigma_x, params.sigma_y)

    # Step: Quantize each pixel based on threshold
    # quant_mask = quantize(smooth_gray, params.quant_threshold)
    quant_mask, threshold = binarize(smooth_gray, params.threshold_algorithm, params.threshold_info)

    # Step: Overlay mask on original color image
    final_overlay = overlay_mask_on_color(
        input_color,
        quant_mask,
        overlay_color=params.overlay_color,
        use_zero=params.overlay_use_zero,
    )

    # Histograms for displays
    hist_input_gray = compute_histogram(gray)
    hist_eq_gray = compute_histogram(eq_gray)
    hist_smooth_gray = compute_histogram(smooth_gray)
    # hist_quant = compute_histogram(quant_mask)

    # Stats
    total_pixels = quant_mask.size
    count_0 = np.sum(~quant_mask)
    percent_0 = (count_0 / total_pixels * 100.0) if total_pixels > 0 else 0.0

    stats = {
        "hist_input_gray": hist_input_gray,
        "hist_eq_gray": hist_eq_gray,
        "hist_smooth_gray": hist_smooth_gray,
        # "hist_quant": hist_quant,
        "count_0": count_0,
        "percent_0": percent_0,
        "total_pixels": total_pixels,
        "threshold_used": threshold,
    }

    return input_color, gray, eq_gray, smooth_gray, quant_mask, final_overlay, stats


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Step: Open image
    Load an image from disk as BGR using OpenCV. Returns None if load fails.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def save_image(path: str, img: np.ndarray) -> bool:
    """
    Save a BGR image to disk. Returns True on success.
    """
    try:
        return bool(cv2.imwrite(path, img))
    except Exception:
        return False
