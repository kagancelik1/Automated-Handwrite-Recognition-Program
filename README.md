# Handwriting Recognition Web Application

A Flask-based web application that uses a trained deep learning model to recognize handwritten text from uploaded images.

## Features

- 🖋️ **Handwriting Recognition**: Upload images of handwritten text and get digital text output
- 🎨 **Beautiful UI**: Modern, responsive web interface with drag-and-drop functionality
- 📊 **Confidence Scores**: Get confidence metrics for recognition accuracy
- 📁 **Multiple Formats**: Supports PNG, JPG, JPEG, GIF, and BMP image formats
- 💾 **Download Results**: Copy or download recognized text
- 📱 **Mobile Friendly**: Responsive design works on all devices

## Requirements

- Python 3.8 or higher
- The trained model file: `best_model_improved.keras`
- Dependencies listed in `requirements.txt`

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /path/to/your/project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the model file exists**
   Make sure `best_model_improved.keras` is in the same directory as `app.py`

## Running the Application

1. **Start the Flask server**
 Activate the correct virtual environment like so: .\venv\Scripts\Activate.ps1
 Run the program by entering the command line :python app.py
 
2. **Open your web browser**
   Navigate by double clicking to in terminal: `http://localhost:5000`

3. **Upload and analyze images**
   - Click "Choose Image File" or drag and drop an image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Maximum file size: 16MB
   - Click "Analyze Handwriting" to get results

## Usage Tips

- **Best Results**: Use high-contrast images with dark text on light backgrounds
- **Image Quality**: Ensure handwriting is clear and not blurry
- **File Size**: Keep images under 16MB for optimal performance
- **Format**: The model works best with single lines of handwritten text

## Project Structure

```
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── best_model_improved.keras   # Trained model file
├── templates/                  # HTML templates
│   ├── index.html             # Upload page
│   └── result.html            # Results page
├── uploads/                   # Uploaded images (created automatically)
└── README.md                  # This file
```

## API Endpoints

- `GET /` - Main upload page
- `POST /predict` - Handle file upload and prediction
- `GET /uploads/<filename>` - Serve uploaded files
- Error handling for file size limits and invalid formats

## Technical Details

- **Model**: Uses `best_model_improved.keras` trained for handwriting recognition
- **Preprocessing**: Images are resized to 256x32 pixels and normalized
- **CTC Decoding**: Uses TensorFlow's CTC decoder for text extraction
- **Character Set**: Supports letters, numbers, and common punctuation

## Troubleshooting

**Model not found error:**
- Ensure `best_model_improved.keras` is in the same directory as `app.py`

**Import errors:**
- Install all dependencies: `pip install -r requirements.txt`

**Port already in use:**
- The app runs on port 5000 by default
- Change the port in `app.py` if needed: `app.run(port=5001)`

**Poor recognition results:**
- Try images with better contrast
- Ensure handwriting is clear and legible
- Use images with dark text on light backgrounds

## Development

To modify the application:

1. **Change model**: Replace `best_model_improved.keras` with your model
2. **Modify UI**: Edit templates in the `templates/` folder
3. **Adjust preprocessing**: Modify the `preprocess_image()` function in `app.py`
4. **Add features**: Extend the Flask routes and templates as needed

## License

This project is for educational and research purposes. 