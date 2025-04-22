# FWHM Calculator

A Python application to calculate the Full Width at Half Maximum (FWHM) of stars in astronomical images (FITS or TIFF formats). It features a user-friendly GUI, interactive star selection, and adaptive image stretching for enhanced visualization, making it ideal for assessing telescope focus and image quality.

## Features
- Supports FITS (.fits, .fit, .fts) and TIFF (.tif, .tiff) images (8/16/32-bit)
- Adaptive image stretching (linear, sqrt, log, asinh, histogram equalization)
- 2D Gaussian fitting to measure FWHM in X and Y directions
- Visualizes results with star position and zoomed-in star profile
- Cross-platform GUI built with Tkinter

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/astrotaca/fwhm-calculator.git
   cd fwhm-calculator
   ```

2. **Install dependencies**:
   Ensure Python 3.7+ is installed, then install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install:
   ```bash
   pip install numpy astropy pillow scipy matplotlib scikit-image
   ```

3. **Optional dependencies** for enhanced TIFF support and performance:
   ```bash
   pip install numba tifffile opencv-python
   ```

## Usage

1. **Run the application**:
   ```bash
   python fwhm_calculator.py
   ```

2. **Steps**:
   - Click "Open Image" to select a FITS or TIFF file.
   - In the image window, click a star to select it (scroll to zoom, press Esc to cancel).
   - View the FWHM results in a pop-up window, including star position, FWHM in X and Y, and average FWHM.
   - Check the Matplotlib plots for a visualization of the star and its fitted profile.

**Tip**: Choose a bright, unsaturated star for accurate results.

## Dependencies

- **Required**:
  - `numpy>=1.21`: Array operations and image processing
  - `astropy>=5.0`: FITS file reading
  - `pillow>=9.0`: TIFF file loading and grayscale conversion
  - `scipy>=1.7`: Gaussian fitting and star detection
  - `matplotlib>=3.5`: Interactive visualization
  - `scikit-image>=0.19`: Adaptive image stretching

- **Optional**:
  - `numba>=0.55`: Optimizes Gaussian fitting
  - `tifffile>=2022.7`: Advanced TIFF support
  - `opencv-python>=4.5`: Alternative TIFF loading

- **Built-in**: `tkinter`, `os`, `sys`, `traceback`, `font`

## Supported Formats
- FITS: `.fits`, `.fit`, `.fts`
- TIFF: `.tif`, `.tiff` (8/16/32-bit, grayscale, RGB, RGBA)

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please follow the [contributing guidelines](CONTRIBUTING.md) and ensure code style consistency (e.g., use `black` for formatting).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, contact [astrotaca](https://github.com/astrotaca) via GitHub Issues.
