"""
image_utils.py - Utility functions for loading and processing astronomical images
"""
import os
import traceback
import numpy as np
from astropy.io import fits
from PIL import Image, ImageOps

def load_image(file_path):
    """Load FITS or TIF file and return image data, handling various bit depths."""
    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Normalize file path
        file_path = os.path.normpath(file_path)
        print(f"Loading file: {file_path}")
        
        if file_path.lower().endswith(('.fits', '.fit', '.fts')):
            # Reading FITS files using Astropy
            try:
                with fits.open(file_path) as hdul:
                    image_data = hdul[0].data
                    # If image is 3D (has multiple planes), take the first plane
                    if len(image_data.shape) > 2:
                        image_data = image_data[0]
                    print(f"FITS data shape: {image_data.shape}, dtype: {image_data.dtype}")
                    return image_data
            except Exception as e:
                print(f"Error loading FITS file: {e}")
                traceback.print_exc()
                raise
                
        elif file_path.lower().endswith(('.tif', '.tiff')):
            # Try multiple methods to load TIFF files
            
            # First attempt: using PIL
            try:
                print("Attempting to load TIFF with PIL...")
                img = Image.open(file_path)
                print(f"Opened image with format: {img.format}, mode: {img.mode}, size: {img.size}")
                
                # Handle different image modes
                if img.mode == 'I':  # 32-bit signed integer
                    print("Processing 32-bit integer image")
                    image_data = np.array(img)
                elif img.mode == 'F':  # 32-bit float
                    print("Processing 32-bit float image")
                    image_data = np.array(img)
                elif img.mode == 'I;16':  # 16-bit unsigned integer
                    print("Processing 16-bit unsigned integer image")
                    image_data = np.array(img)
                elif img.mode == 'RGB' or img.mode == 'RGBA':
                    print(f"Converting {img.mode} image to grayscale using luminosity method")
                    # Properly align RGB channels and convert to grayscale using luminosity method
                    # This gives proper weighting to each channel (R:0.2989, G:0.5870, B:0.1140)
                    img_array = np.array(img)
                    if img.mode == 'RGBA':  # Handle alpha channel if present
                        img_array = img_array[:,:,:3]
                    # Apply luminosity method for RGB to grayscale conversion
                    image_data = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    # For other modes, convert to grayscale if needed
                    if img.mode != 'L':
                        print(f"Converting {img.mode} to grayscale")
                        img = ImageOps.grayscale(img)
                    image_data = np.array(img)
                
                print(f"Image data shape: {image_data.shape}, dtype: {image_data.dtype}, min: {np.min(image_data)}, max: {np.max(image_data)}")
                return image_data
                
            except Exception as e:
                print(f"Failed to open TIFF file with PIL: {e}")
                traceback.print_exc()
                
                # Second attempt: try using tifffile if available
                try:
                    import tifffile
                    print("Attempting to load TIFF with tifffile...")
                    image_data = tifffile.imread(file_path)
                    
                    # If image is 3D (has multiple planes), use luminosity method for RGB
                    if len(image_data.shape) > 2:
                        if len(image_data.shape) == 3 and image_data.shape[2] in [3, 4]:  # RGB or RGBA
                            # Use luminosity method for proper RGB to grayscale conversion
                            rgb_data = image_data[...,:3]
                            image_data = np.dot(rgb_data, [0.2989, 0.5870, 0.1140])
                        else:
                            image_data = image_data[0]  # Take first channel/plane
                    
                    print(f"tifffile loaded data shape: {image_data.shape}, dtype: {image_data.dtype}")
                    return image_data
                    
                except ImportError:
                    print("tifffile module not available. Please install it with 'pip install tifffile'")
                except Exception as e2:
                    print(f"Failed to open TIFF with tifffile: {e2}")
                    traceback.print_exc()
                    
                # Third attempt: try using OpenCV if available
                try:
                    import cv2
                    print("Attempting to load TIFF with OpenCV...")
                    # OpenCV reads images as BGR, need to handle conversion properly
                    image_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    
                    # Check if image was loaded
                    if image_data is None:
                        raise IOError("OpenCV returned None when loading the image")
                        
                    # Convert BGR to grayscale if it's a color image
                    if len(image_data.shape) > 2 and image_data.shape[2] >= 3:
                        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                        
                    print(f"OpenCV loaded data shape: {image_data.shape}, dtype: {image_data.dtype}")
                    return image_data
                    
                except ImportError:
                    print("OpenCV module not available. Please install it with 'pip install opencv-python'")
                except Exception as e3:
                    print(f"Failed to open TIFF with OpenCV: {e3}")
                    traceback.print_exc()
                    
                # If all attempts failed, raise the original error
                raise IOError(f"Failed to open TIFF file. Tried multiple methods but all failed. Original error: {e}")
        else:
            raise ValueError("Unsupported file format. Please provide a .fits, .fit, .tif, or .tiff file.")
            
    except Exception as e:
        print(f"Error loading image: {e}")
        traceback.print_exc()
        raise

def stretch_image(image_data):
    """Basic linear stretching function for displaying images."""
    import numpy as np
    
    # Convert to float for processing
    float_data = image_data.astype(np.float64)
    
    # Handle potential NaN or inf values
    float_data = np.nan_to_num(float_data)
    
    min_val = np.min(float_data)
    max_val = np.max(float_data)
    
    if max_val == min_val:  # Avoid division by zero
        print("Warning: Image has no contrast (min == max)")
        return np.zeros_like(image_data, dtype=np.uint8)
    
    # Scale to 0-255 range
    stretched_image = ((float_data - min_val) / (max_val - min_val) * 255.0)
    
    # Clip values to ensure they're in 0-255 range
    stretched_image = np.clip(stretched_image, 0, 255)
    
    return stretched_image.astype(np.uint8)

def adaptive_stretch_image(image_data, method="auto"):
    """
    Apply adaptive stretching to enhance image contrast based on image characteristics.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        The input image data
    method : str
        Stretching method: "auto", "linear", "sqrt", "log", "asinh", "histogram"
        If "auto", the best method will be selected based on image statistics
        
    Returns:
    --------
    numpy.ndarray
        Stretched image data (0-255 range, uint8 type)
    """
    try:
        from skimage import exposure
    except ImportError:
        print("scikit-image not available. Using basic stretching instead.")
        return stretch_image(image_data)
    
    # Convert to float for processing
    float_data = image_data.astype(np.float64)
    
    # Handle potential NaN or inf values
    float_data = np.nan_to_num(float_data)
    
    # Get image statistics
    min_val = np.min(float_data)
    max_val = np.max(float_data)
    mean_val = np.mean(float_data)
    median_val = np.median(float_data)
    std_val = np.std(float_data)
    
    # Detect if image is low contrast
    dynamic_range = max_val - min_val
    if dynamic_range == 0:  # Avoid division by zero
        print("Warning: Image has no contrast (min == max)")
        return np.zeros_like(image_data, dtype=np.uint8)
    
    # Calculate contrast metrics
    contrast_ratio = dynamic_range / mean_val if mean_val > 0 else 0
    signal_to_noise = mean_val / std_val if std_val > 0 else 0
    
    # Auto-select the best stretching method based on image statistics
    if method == "auto":
        # Analyze histogram to detect image type
        hist, bins = np.histogram(float_data.flatten(), bins=256)
        hist_peak = np.argmax(hist)
        hist_peak_ratio = hist_peak / 256
        
        # Calculate skewness to detect bright outliers (like stars)
        diff = float_data - mean_val
        skewness = np.mean(diff**3) / (std_val**3) if std_val > 0 else 0
        
        print(f"Image statistics - Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}, Median: {median_val:.2f}")
        print(f"Contrast ratio: {contrast_ratio:.2f}, SNR: {signal_to_noise:.2f}, Skewness: {skewness:.2f}")
        
        # Decision logic for automatic method selection
        if skewness > 3.0:  # Strong positive skew (bright outliers like stars)
            print("Detected bright outliers - using asinh stretch")
            method = "asinh"
        elif contrast_ratio < 0.5:  # Low contrast images
            print("Detected low contrast - using histogram equalization")
            method = "histogram"
        elif hist_peak_ratio < 0.2:  # Dark image with few bright details
            print("Detected dark image - using sqrt stretch")
            method = "sqrt"
        elif signal_to_noise < 2.0:  # Noisy image
            print("Detected noisy image - using log stretch")
            method = "log"
        else:  # Default to linear stretch with percentile clipping
            print("Using linear stretch with percentile clipping")
            method = "linear"
    
    # Apply the selected stretching method
    if method == "linear":
        # Use percentile clipping to remove outliers
        p_low, p_high = 1, 99
        low = np.percentile(float_data, p_low)
        high = np.percentile(float_data, p_high)
        
        # Clip and scale to 0-1 range
        clipped = np.clip(float_data, low, high)
        normalized = (clipped - low) / (high - low) if high > low else clipped
        stretched = normalized * 255.0
        
    elif method == "sqrt":
        # Square root stretch - good for images with faint detail
        normalized = (float_data - min_val) / dynamic_range if dynamic_range > 0 else float_data
        stretched = np.sqrt(normalized) * 255.0
        
    elif method == "log":
        # Logarithmic stretch - good for high dynamic range
        # Add small constant to avoid log(0)
        log_data = np.log1p(float_data - min_val)
        stretched = (log_data / np.max(log_data) if np.max(log_data) > 0 else log_data) * 255.0
        
    elif method == "asinh":
        # Arcsinh stretch - especially good for astronomical images
        # Similar to log but with better behavior near zero
        asinh_data = np.arcsinh((float_data - min_val) / (std_val * 0.1))
        stretched = (asinh_data / np.max(asinh_data) if np.max(asinh_data) > 0 else asinh_data) * 255.0
        
    elif method == "histogram":
        # Adaptive histogram equalization - best for bringing out local details
        # Convert to 0-1 range first
        normalized = (float_data - min_val) / dynamic_range if dynamic_range > 0 else float_data
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        try:
            # Try skimage's exposure module for CLAHE
            stretched = exposure.equalize_adapthist(normalized, clip_limit=0.03) * 255.0
        except:
            # Fallback to simple histogram equalization
            stretched = exposure.equalize_hist(normalized) * 255.0
            
    else:
        raise ValueError(f"Unknown stretching method: {method}")
    
    # Ensure values are in 0-255 range
    stretched = np.clip(stretched, 0, 255)
    
    return stretched.astype(np.uint8)

def compare_stretching_methods(image_data):
    """
    Compare different stretching methods and display results side by side.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        The input image data
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    methods = ["linear", "sqrt", "log", "asinh", "histogram"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(stretch_image(image_data), cmap='gray', origin='lower')
    axes[0, 0].set_title("Original Stretch")
    
    # Auto-selected method
    auto_stretched = adaptive_stretch_image(image_data, method="auto")
    axes[0, 1].imshow(auto_stretched, cmap='gray', origin='lower')
    axes[0, 1].set_title("Auto-Selected Method")
    
    # Empty placeholder
    axes[0, 2].axis('off')
    
    # Other methods
    for i, method in enumerate(methods):
        row, col = (i // 3) + 1, i % 3
        stretched = adaptive_stretch_image(image_data, method=method)
        axes[row, col].imshow(stretched, cmap='gray', origin='lower')
        axes[row, col].set_title(f"{method.capitalize()} Stretch")
    
    plt.tight_layout()
    plt.show()
    
    return auto_stretched