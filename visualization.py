import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox
import os

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

def compare_stretching_methods(image_data):
    """
    Compare different stretching methods and display results side by side.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        The input image data
    
    Returns:
    --------
    numpy.ndarray
        The auto-stretched image data
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

def stretch_image(image_data):
    """
    Original basic stretching function for comparison.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        The input image data
        
    Returns:
    --------
    numpy.ndarray
        Stretched image data (0-255 range, uint8 type)
    """
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

def interactive_star_selection(image_data):
    """
    Display image and allow user to select a star interactively.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        The image data to display
        
    Returns:
    --------
    tuple or None
        (x, y) coordinates of selected star or None if no selection was made
    """
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

def display_results(result_data, file_path):
    """
    Display the FWHM results in a window and plot visualization.
    
    Parameters:
    -----------
    result_data : tuple
        (FWHM_x, FWHM_y, fitted_x, fitted_y, image_data, stretched_image)
    file_path : str
        Path to the image file that was analyzed
    """
    if result_data is None:
        return
        
    FWHM_x, FWHM_y, fitted_x, fitted_y, image_data, stretched_image = result_data
    
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
    visualize_star_fit(stretched_image, fitted_x, fitted_y, FWHM_x, FWHM_y)

def visualize_star_fit(stretched_image, fitted_x, fitted_y, FWHM_x, FWHM_y):
    """
    Create visualization of the star fit results.
    
    Parameters:
    -----------
    stretched_image : numpy.ndarray
        Stretched image data for display
    fitted_x, fitted_y : float
        Fitted center coordinates of the star
    FWHM_x, FWHM_y : float
        FWHM values in x and y directions
    """
    try:
        # Validate stretched_image
        if stretched_image is None or not isinstance(stretched_image, np.ndarray) or stretched_image.ndim != 2 or stretched_image.size == 0:
            raise ValueError("Invalid stretched image data for plotting")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

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
        print(f"Error in visualization: {e}")
        messagebox.showerror("Visualization Error", f"Could not visualize results: {str(e)}")
