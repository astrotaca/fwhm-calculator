"""
main.py - Main application file for FWHM Calculator
Contains the GUI implementation and application entry point
"""
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
from tkinter import filedialog
from tkinter import messagebox
import os
import traceback
import sys

# Import our modules
from image_utils import load_image, stretch_image, adaptive_stretch_image
from star_analysis import find_brightest_star, fit_star
from visualization import interactive_star_selection

class FWHMCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("FWHM Calculator")
        self.root.geometry("400x300")
        self.root.configure(bg="#f0f0f0")  # Light, modern background
        
        # Set a clean, readable font
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Arial", size=10)
        
        # Main frame with padding for breathing room
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bold, standout title
        title_label = ttk.Label(self.main_frame, text="FWHM Calculator", font=("Arial", 16, "bold"), background="#f0f0f0")
        title_label.pack(pady=(0, 10))
        
        # Concise, helpful description
        description = """Calculate the Full Width at Half Maximum (FWHM) of stars.

Supported formats: FITS (.fits), TIFF (.tif, .tiff) - 8/16/32-bit

How to use:
1. Click 'Open Image' to load your image.
2. Click a star or drag to zoom in the image window.
3. View the FWHM result.

Tip: Pick a bright, unsaturated star."""
        desc_label = ttk.Label(self.main_frame, text=description, justify=tk.LEFT, background="#f0f0f0")
        desc_label.pack(pady=(0, 20))
        
        # Buttons in a neat frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X)
        open_button = ttk.Button(button_frame, text="Open Image", command=self.open_file_dialog, width=15)
        open_button.pack(side=tk.LEFT, padx=5)
        quit_button = ttk.Button(button_frame, text="Quit", command=self.root.quit, width=15)
        quit_button.pack(side=tk.RIGHT, padx=5)
        
        # Status bar for feedback
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W, background="#e0e0e0")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def open_file_dialog(self):
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
                self.process_image(file_path)
            except Exception as e:
                print(f"Unexpected error: {e}")
                traceback.print_exc()
                messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
    
    def process_image(self, file_path):
        """Main function to process the image and calculate FWHM of stars."""
        try:
            print("\n--- Starting image processing ---")
            # Update status
            self.status_bar.config(text="Loading image...")
            self.root.update()
            
            # Load the image
            image_data = load_image(file_path)
            
            # Update status
            self.status_bar.config(text="Enhancing image...")
            self.root.update()
            
            # Stretch for display
            stretched_image = stretch_image(image_data)
            
            # Update status
            self.status_bar.config(text="Detecting stars...")
            self.root.update()
            
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
            self.status_bar.config(text="Fitting Gaussian profile...")
            self.root.update()
            
            # Fit the star
            result = fit_star(stretched_image, x_center, y_center)
            if result[0] is None:
                self.status_bar.config(text="Ready")
                return
            
            FWHM_x, FWHM_y, fitted_x, fitted_y = result
            
            # Update status
            self.status_bar.config(text=f"FWHM: X={FWHM_x:.2f}, Y={FWHM_y:.2f} pixels")
            self.root.update()
    
            # Display results
            print(f"Star position: ({fitted_x:.2f}, {fitted_y:.2f})")
            print(f"FWHM in X direction: {FWHM_x:.2f} pixels")
            print(f"FWHM in Y direction: {FWHM_y:.2f} pixels")
            print(f"Average FWHM: {(FWHM_x + FWHM_y) / 2:.2f} pixels")
            
            self.show_results(file_path, image_data, stretched_image, fitted_x, fitted_y, FWHM_x, FWHM_y)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()
            self.status_bar.config(text="Error processing image")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def show_results(self, file_path, image_data, stretched_image, fitted_x, fitted_y, FWHM_x, FWHM_y):
        """Display the results in a new window."""
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
        
        # Import matplotlib inside function to avoid circular imports
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import numpy as np
        
        # Visualize the result
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # First subplot - full image with star marked
        img1 = ax1.imshow(stretched_image, cmap='gray', origin='lower')
        ax1.scatter(fitted_x, fitted_y, color='red', marker='+', s=100, label='Star Center')
        avg_fwhm = (FWHM_x + FWHM_y) / 2
        circle = plt.Circle((fitted_x, fitted_y), avg_fwhm/2, color='r', fill=False)
        ax1.add_patch(circle)
        fig.colorbar(img1, ax=ax1, label='Intensity')
        ax1.legend()
        ax1.set_title('Star Detection')
        
        # Second subplot - zoomed in view of the star
        size = int(max(FWHM_x, FWHM_y) * 4)
        size = max(size, 10)  # Ensure minimum size
        x_min = max(0, int(fitted_x) - size)
        x_max = min(stretched_image.shape[1], int(fitted_x) + size)
        y_min = max(0, int(fitted_y) - size)
        y_max = min(stretched_image.shape[0], int(fitted_y) + size)
        zoom_region = stretched_image[y_min:y_max, x_min:x_max]
        
        img2 = ax2.imshow(zoom_region, cmap='gray', origin='lower')
        ax2.scatter(fitted_x - x_min, fitted_y - y_min, color='red', marker='+', s=100)
        fig.colorbar(img2, ax=ax2, label='Intensity')
        ax2.set_title('Zoom on Star')
        
        fig.tight_layout()
        
        # Embed plot in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def main():
    # Set up better exception handling for the entire application
    def show_error(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(f"Uncaught exception: {error_msg}")
        messagebox.showerror("Error", f"An unexpected error occurred:\n{str(exc_value)}")
    
    # Set the exception hook
    sys.excepthook = show_error
    
    # Start the application
    root = tk.Tk()
    app = FWHMCalculator(root)
    root.mainloop()

if __name__ == "__main__":
    main()