# Image Classification with VGG16

A Python application that performs image classification using the pre-trained VGG16 deep learning model. The application can recognize up to 1000 different object categories from the ImageNet dataset.

## Features

- Classify images using the powerful VGG16 model
- Displays top 3 predictions with confidence scores
- Visualizes results with a clean interface
- Command-line interface for easy integration
- Option to run in headless mode (without displaying images)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/image-classification.git
   cd image-classification
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python image_classifier.py path/to/your/image.jpg
```

### Headless Mode (without displaying images)

```bash
python image_classifier.py path/to/your/image.jpg --no-display
```

### Example Output

```
Loading VGG16 model (this may take a moment)...
Classifying image: examples/elephant.jpg

Top 3 Predictions:
--------------------------------------------------
1. African Elephant: 92.14%
2. Tusker: 7.39%
3. Indian Elephant: 0.47%
```

## Example Images

We've included some example images in the `examples/` directory that you can use to test the classifier.

## Requirements

- Python 3.7+
- TensorFlow 2.10.0 or higher
- NumPy
- Matplotlib
- Pillow (PIL Fork)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The VGG16 model is pre-trained on the ImageNet dataset
- Built using TensorFlow and Keras
- Inspired by various computer vision tutorials and examples
