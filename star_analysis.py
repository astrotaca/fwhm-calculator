import numpy as np
from numba import jit
from scipy.optimize import curve_fit
from tkinter import messagebox
import traceback
import os

@jit(nopython=True)
def gaussian_function(xy, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function for star fitting.
    
    Parameters:
    -----------
    xy : tuple of arrays
        x and y coordinates
    amp : float
        Amplitude of the Gaussian
    x0, y0 : float
        Center coordinates of the Gaussian
    sigma_x, sigma_y : float
        Standard deviations in x and y directions
    theta : float
        Rotation angle of the Gaussian
    offset : float
        Baseline offset
        
    Returns:
    --------
    float
        Value of the Gaussian function at the specified points
    """
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    return amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2)) + offset

def fit_star(image_data, x_center, y_center, size=20):
    """
    Fit a 2D Gaussian to a star to determine its FWHM.
    
    Parameters:
    -----------
    image_data : ndarray
        The image data containing the star
    x_center, y_center : float
        Approximate center coordinates of the star
    size : int
        Size of the region around the star to use for fitting
        
    Returns:
    --------
    tuple
        (FWHM_x, FWHM_y, fitted_x_center, fitted_y_center) or (None, None, None, None) if fit fails
    """
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
    """
    Find the brightest star in the image using thresholding and centroiding.
    
    Parameters:
    -----------
    image_data : ndarray
        The image data to analyze
    threshold_percentile : float
        Percentile threshold for star detection
        
    Returns:
    --------
    tuple
        (x_center, y_center) coordinates of the brightest star
    """
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

def process_image(file_path, load_image, stretch_image, interactive_star_selection, status_bar=None, root=None):
    """
    Main function to process the image and calculate FWHM of stars.
    
    Parameters:
    -----------
    file_path : str
        Path to the image file
    load_image : function
        Function to load the image file
    stretch_image : function
        Function to stretch the image for display
    interactive_star_selection : function
        Function for interactive star selection
    status_bar : tkinter.Label, optional
        Status bar widget for updating status
    root : tkinter.Tk, optional
        Root window for updating the GUI
        
    Returns:
    --------
    tuple
        Results of the star fitting (FWHM_x, FWHM_y, fitted_x, fitted_y)
    """
    try:
        print("\n--- Starting image processing ---")
        # Update status
        if status_bar is not None and root is not None:
            status_bar.config(text="Loading image...")
            root.update()
        
        # Load the image
        image_data = load_image(file_path)
        
        # Update status
        if status_bar is not None and root is not None:
            status_bar.config(text="Enhancing image...")
            root.update()
        
        # Stretch for display
        stretched_image = stretch_image(image_data)
        
        # Update status
        if status_bar is not None and root is not None:
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
        if status_bar is not None and root is not None:
            status_bar.config(text="Fitting Gaussian profile...")
            root.update()
        
        # Fit the star
        result = fit_star(stretched_image, x_center, y_center)
        if result[0] is None:
            if status_bar is not None:
                status_bar.config(text="Ready")
            return None
        
        FWHM_x, FWHM_y, fitted_x, fitted_y = result
        
        # Update status
        if status_bar is not None and root is not None:
            status_bar.config(text=f"FWHM: X={FWHM_x:.2f}, Y={FWHM_y:.2f} pixels")
            root.update()

        # Display results
        print(f"Star position: ({fitted_x:.2f}, {fitted_y:.2f})")
        print(f"FWHM in X direction: {FWHM_x:.2f} pixels")
        print(f"FWHM in Y direction: {FWHM_y:.2f} pixels")
        print(f"Average FWHM: {(FWHM_x + FWHM_y) / 2:.2f} pixels")
        
        return (FWHM_x, FWHM_y, fitted_x, fitted_y, image_data, stretched_image)

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        if status_bar is not None:
            status_bar.config(text="Error processing image")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        return None