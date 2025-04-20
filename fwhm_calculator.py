import numpy as np
from astropy.io import fits
from PIL import Image, ImageOps
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
from tkinter import filedialog
from tkinter import messagebox
import os
import traceback
import sys
from numba import jit

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
                    messagebox.showinfo("Dependency Required", 
                                      "The 'tifffile' module is required for advanced TIFF support.\n"
                                      "Please install it with:\npip install tifffile")
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
                    messagebox.showinfo("Dependency Required", 
                                      "The 'opencv-python' module is required for advanced image support.\n"
                                      "Please install it with:\npip install opencv-python")
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
        messagebox.showerror("Error", f"An error occurred while loading the image: {e}")
        raise

    
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
    import numpy as np
    from skimage import exposure
    
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

# Function to visualize the effects of different stretching methods
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

# Improved version of the original stretch_image function
def stretch_image(image_data):
    """Original basic stretching function for comparison."""
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

@jit(nopython=True)
def gaussian_function(xy, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    return amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2)) + offset

def fit_star(image_data, x_center, y_center, size=20):
    x_center, y_center = int(x_center), int(y_center)
    height, width = image_data.shape
    x_min = max(0, x_center - size)
    x_max = min(width, x_center + size)
    y_min = max(0, y_center - size)
    y_max = min(height, y_center + size)
    star_region = image_data[y_min:y_max, x_min:x_max]

    if star_region.size == 0:
        raise ValueError("Selected region is empty")

    # Check for saturation
    if np.issubdtype(image_data.dtype, np.integer) and np.max(star_region) >= np.iinfo(image_data.dtype).max:
        messagebox.showwarning("Warning", "Star might be saturated. Results may be off.")

    y_grid, x_grid = np.mgrid[0:star_region.shape[0], 0:star_region.shape[1]]
    xy_data = np.vstack((x_grid.ravel(), y_grid.ravel()))
    z_data = star_region.ravel()

    # Smarter initial guesses
    max_val = np.max(star_region)
    min_val = np.min(star_region)
    half_max = (max_val + min_val) / 2
    above_half = star_region > half_max
    sigma_x_guess = np.std(np.where(above_half)[1]) if np.any(above_half) else 2.0
    sigma_y_guess = np.std(np.where(above_half)[0]) if np.any(above_half) else 2.0
    initial_guess = (max_val - min_val, star_region.shape[1] // 2, star_region.shape[0] // 2, sigma_x_guess, sigma_y_guess, 0, min_val)

    bounds = ([0, 0, 0, 0.1, 0.1, -np.pi/2, 0], [np.inf, star_region.shape[1], star_region.shape[0], size, size, np.pi/2, np.inf])

    try:
        popt, _ = curve_fit(gaussian_function, xy_data, z_data, p0=initial_guess, bounds=bounds)
        amp, x0, y0, sigma_x, sigma_y, theta, offset = popt
        FWHM_x = 2.355 * sigma_x
        FWHM_y = 2.355 * sigma_y

        # Check if the fit makes sense
        if sigma_x < 1 or sigma_y < 1:
            messagebox.showwarning("Warning", "Fit might be poor (sigma too small). Try another star.")

        return FWHM_x, FWHM_y, x_min + x0, y_min + y0
    except Exception as e:
        messagebox.showerror("Error", f"Fit failed: {e}")
        return None, None, None, None

def find_brightest_star(image_data, threshold_percentile=99.9):
    """Find the brightest star in the image using thresholding and centroiding."""
    # Apply a threshold to identify potential stars
    threshold = np.percentile(image_data, threshold_percentile)
    binary_img = image_data > threshold
    
    # Label connected components (potential stars)
    from scipy import ndimage
    labeled_img, num_features = ndimage.label(binary_img)
    
    if num_features == 0:
        print("No stars found. Using brightest pixel instead.")
        y_center, x_center = np.unravel_index(np.argmax(image_data), image_data.shape)
        return x_center, y_center
    
    # Find properties of detected regions
    regions = ndimage.find_objects(labeled_img)
    
    # Find the brightest region
    max_intensity = 0
    brightest_star_center = None
    
    for i, region in enumerate(regions):
        if region is not None:
            # Create a mask for this region
            mask = labeled_img[region] == i + 1
            # Extract the star region
            star_region = image_data[region] * mask
            # Find the maximum intensity
            region_max = np.max(star_region)
            
            if region_max > max_intensity:
                max_intensity = region_max
                # Calculate centroid
                y_indices, x_indices = np.nonzero(mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    # Get region origin
                    y_origin = region[0].start
                    x_origin = region[1].start
                    # Calculate intensity-weighted centroid
                    weights = star_region[y_indices, x_indices]
                    x_centroid = np.sum(x_indices * weights) / np.sum(weights) + x_origin
                    y_centroid = np.sum(y_indices * weights) / np.sum(weights) + y_origin
                    brightest_star_center = (x_centroid, y_centroid)
    
    if brightest_star_center is None:
        # Fallback to brightest pixel
        print("Centroid calculation failed. Using brightest pixel instead.")
        y_center, x_center = np.unravel_index(np.argmax(image_data), image_data.shape)
        return x_center, y_center
    
    return brightest_star_center

import matplotlib.pyplot as plt
import numpy as np

def interactive_star_selection(image_data):
    global selected_point
    selected_point = None

    # Validate input data
    if image_data is None or not isinstance(image_data, np.ndarray) or image_data.ndim != 2 or image_data.size == 0:
        raise ValueError("Invalid image data for plotting")

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(image_data, cmap='gray', origin='lower')
    fig.colorbar(img, ax=ax, label='Intensity')
    ax.set_title('Select a Star')

    # Initialize zoom parameters
    zoom_factor = 1.0  # Current zoom level (1.0 = original size)
    base_xlim = ax.get_xlim()
    base_ylim = ax.get_ylim()
    image_height, image_width = image_data.shape

    # Guide the user
    ax.text(0.5, -0.1, "Click a star or scroll to zoom. Press 'Esc' to cancel.",
            horizontalalignment='center', transform=ax.transAxes, fontsize=10, color='red')

    def find_nearest_bright_spot(x, y, search_radius=10):
        x, y = int(x), int(y)
        height, width = image_data.shape
        x_min = max(0, x - search_radius)
        x_max = min(width, x + search_radius + 1)
        y_min = max(0, y - search_radius)
        y_max = min(height, y + search_radius + 1)
        region = image_data[y_min:y_max, x_min:x_max]
        if region.size == 0:
            return x, y
        local_y, local_x = np.unravel_index(np.argmax(region), region.shape)
        return x_min + local_x, y_min + local_y

    def onclick(event):
        global selected_point
        if event.xdata is not None and event.ydata is not None:
            x, y = find_nearest_bright_spot(event.xdata, event.ydata)
            selected_point = (x, y)
            plt.close()

    def onscroll(event):
        if event.xdata is None or event.ydata is None:
            return

        # Get current mouse position
        x, y = event.xdata, event.ydata

        # Zoom in (scroll up) or out (scroll down)
        if event.button == 'up':
            zoom_factor_new = zoom_factor * 1.2  # Zoom in by 20%
        elif event.button == 'down':
            zoom_factor_new = zoom_factor / 1.2  # Zoom out by 20%
        else:
            return

        # Limit zoom to reasonable bounds
        zoom_factor_new = max(1.0, min(zoom_factor_new, 10.0))  # Min: 1x, Max: 10x
        if zoom_factor_new == zoom_factor:
            return

        # Calculate new limits centered on mouse position
        scale = zoom_factor / zoom_factor_new
        new_width = (base_xlim[1] - base_xlim[0]) * scale
        new_height = (base_ylim[1] - base_ylim[0]) * scale

        x_left = x - (x - base_xlim[0]) * scale
        x_right = x_left + new_width
        y_bottom = y - (y - base_ylim[0]) * scale
        y_top = y_bottom + new_height

        # Ensure limits stay within image bounds
        x_left = max(0, min(x_left, image_width - new_width))
        x_right = x_left + new_width
        y_bottom = max(0, min(y_bottom, image_height - new_height))
        y_top = y_bottom + new_height

        # Update axes limits
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)
        zoom_factor = zoom_factor_new

        fig.canvas.draw_idle()

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('scroll_event', onscroll)

    # Handle Esc key to cancel
    def onkey(event):
        if event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    return selected_point

def process_image(file_path):
    """Main function to process the image and calculate FWHM of stars."""
    try:
        print("\n--- Starting image processing ---")
        # Update status
        if 'status_bar' in globals():
            status_bar.config(text="Loading image...")
            root.update()
        
        # Load the image
        image_data = load_image(file_path)
        
        # Update status
        if 'status_bar' in globals():
            status_bar.config(text="Enhancing image...")
            root.update()
        
        # Stretch for display
        stretched_image = stretch_image(image_data)
        
        # Update status
        if 'status_bar' in globals():
            status_bar.config(text="Detecting stars...")
            root.update()
        
        # Display the image and let user select a star
        selection = interactive_star_selection(stretched_image)
        
        if selection is None:
            print("No star selected. Detecting brightest star automatically.")
            # Find brightest star
            x_center, y_center = find_brightest_star(stretched_image)
        else:
            x_center, y_center = selection
        
        print(f"Selected star position: ({x_center:.2f}, {y_center:.2f})")
        
        # Update status
        if 'status_bar' in globals():
            status_bar.config(text="Fitting Gaussian profile...")
            root.update()
        
        # Fit the star
        result = fit_star(stretched_image, x_center, y_center)
        if result[0] is None:
            if 'status_bar' in globals():
                status_bar.config(text="Ready")
            return
        
        FWHM_x, FWHM_y, fitted_x, fitted_y = result
        
        # Update status
        if 'status_bar' in globals():
            status_bar.config(text=f"FWHM: X={FWHM_x:.2f}, Y={FWHM_y:.2f} pixels")
            root.update()

        # Display results
        print(f"Star position: ({fitted_x:.2f}, {fitted_y:.2f})")
        print(f"FWHM in X direction: {FWHM_x:.2f} pixels")
        print(f"FWHM in Y direction: {FWHM_y:.2f} pixels")
        print(f"Average FWHM: {(FWHM_x + FWHM_y) / 2:.2f} pixels")
        
        # Create a result window
        result_window = tk.Toplevel()
        result_window.title("FWHM Results")
        result_frame = tk.Frame(result_window, padx=10, pady=10)
        result_frame.pack()
        
        # File info
        tk.Label(result_frame, text=f"File: {os.path.basename(file_path)}", font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Label(result_frame, text=f"Image size: {image_data.shape[1]}×{image_data.shape[0]} pixels").pack(anchor="w")
        tk.Label(result_frame, text=f"Data type: {image_data.dtype}").pack(anchor="w")
        
        # Empty row
        tk.Label(result_frame, text="").pack()
        
        # Results
        tk.Label(result_frame, text="Results:", font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Label(result_frame, text=f"Star position: ({fitted_x:.2f}, {fitted_y:.2f})").pack(anchor="w")
        tk.Label(result_frame, text=f"FWHM in X direction: {FWHM_x:.2f} pixels").pack(anchor="w")
        tk.Label(result_frame, text=f"FWHM in Y direction: {FWHM_y:.2f} pixels").pack(anchor="w")
        tk.Label(result_frame, text=f"Average FWHM: {(FWHM_x + FWHM_y) / 2:.2f} pixels").pack(anchor="w")
                
        # Visualize the result
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Validate stretched_image
        if stretched_image is None or not isinstance(stretched_image, np.ndarray) or stretched_image.ndim != 2 or stretched_image.size == 0:
            raise ValueError("Invalid stretched image data for plotting")
        img1 = ax1.imshow(stretched_image, cmap='gray', origin='lower')
        ax1.scatter(fitted_x, fitted_y, color='red', marker='+', s=100, label='Star Center')
        avg_fwhm = (FWHM_x + FWHM_y) / 2
        circle = plt.Circle((fitted_x, fitted_y), avg_fwhm/2, color='r', fill=False)
        ax1.add_patch(circle)
        fig.colorbar(img1, ax=ax1, label='Intensity')
        ax1.legend()
        ax1.set_title('Star Detection')

        # Plot a zoom in of the star
        size = int(max(FWHM_x, FWHM_y) * 4)
        size = max(size, 10)  # Ensure minimum size
        x_min = max(0, int(fitted_x) - size)
        x_max = min(stretched_image.shape[1], int(fitted_x) + size)
        y_min = max(0, int(fitted_y) - size)
        y_max = min(stretched_image.shape[0], int(fitted_y) + size)
        zoom_region = stretched_image[y_min:y_max, x_min:x_max]
        if zoom_region is None or not isinstance(zoom_region, np.ndarray) or zoom_region.ndim != 2 or zoom_region.size == 0:
            raise ValueError("Invalid zoomed region data for plotting")
        img2 = ax2.imshow(zoom_region, cmap='gray', origin='lower')
        ax2.scatter(fitted_x - x_min, fitted_y - y_min, color='red', marker='+', s=100)
        fig.colorbar(img2, ax=ax2, label='Intensity')
        ax2.set_title('Zoom on Star')

        fig.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        if 'status_bar' in globals():
            status_bar.config(text="Error processing image")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def open_file_dialog():
    """Open file dialog to select image."""
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("All supported", "*.fits *.fit *.fts *.tif *.tiff"),
            ("FITS files", "*.fits *.fit *.fts"),
            ("TIFF files", "*.tif *.tiff")
        ]
    )
    if file_path:
        try:
            process_image(file_path)
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

import tkinter.ttk as ttk

def create_gui():
    global root, status_bar
    root = tk.Tk()
    root.title("FWHM Calculator")
    root.geometry("400x300")
    root.configure(bg="#f0f0f0")  # Light, modern background

    # Set a clean, readable font
    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(family="Arial", size=10)

    # Main frame with padding for breathing room
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Bold, standout title
    title_label = ttk.Label(main_frame, text="FWHM Calculator", font=("Arial", 16, "bold"), background="#f0f0f0")
    title_label.pack(pady=(0, 10))

    # Concise, helpful description
    description = """Calculate the Full Width at Half Maximum (FWHM) of stars.

Supported formats: FITS (.fits), TIFF (.tif, .tiff) - 8/16/32-bit

How to use:
1. Click 'Open Image' to load your image.
2. Click a star or drag to zoom in the image window.
3. View the FWHM result.

Tip: Pick a bright, unsaturated star."""
    desc_label = ttk.Label(main_frame, text=description, justify=tk.LEFT, background="#f0f0f0")
    desc_label.pack(pady=(0, 20))

    # Buttons in a neat frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X)
    open_button = ttk.Button(button_frame, text="Open Image", command=open_file_dialog, width=15)
    open_button.pack(side=tk.LEFT, padx=5)
    quit_button = ttk.Button(button_frame, text="Quit", command=root.quit, width=15)
    quit_button.pack(side=tk.RIGHT, padx=5)

    # Status bar for feedback
    status_bar = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W, background="#e0e0e0")
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()

if __name__ == "__main__":
    # Set up better exception handling for the entire application
    def show_error(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(f"Uncaught exception: {error_msg}")
        messagebox.showerror("Error", f"An unexpected error occurred:\n{str(exc_value)}")
    
    # Set the exception hook
    sys.excepthook = show_error
    
    # Start the application
    create_gui()