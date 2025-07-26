import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import PIL
import os
import argparse

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for VGG16 model
    
    Args:
        img_path (str): Path to the image file
        target_size (tuple): Target size of the image (width, height)
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
        
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def classify_image(model, img_path):
    """
    Classify an image using the VGG16 model
    
    Args:
        model: Loaded VGG16 model
        img_path (str): Path to the image file
        
    Returns:
        list: List of dictionaries containing top 3 predictions with ranks, labels, and scores
    """
    try:
        # Load and preprocess the image
        img_array = load_and_preprocess_image(img_path)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Decode and return top 3 predictions
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Format the results
        return [
            {
                'rank': i + 1,
                'label': label.replace('_', ' ').title(),
                'score': f"{score * 100:.2f}%"
            }
            for i, (_, label, score) in enumerate(decoded_predictions)
        ]
    except Exception as e:
        raise Exception(f"Error in image classification: {str(e)}")

def display_results(image_path, predictions):
    """Display the input image and prediction results"""
    plt.figure(figsize=(10, 8))
    
    # Display image
    plt.subplot(2, 1, 1)
    img = PIL.Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image')
    
    # Display predictions
    plt.subplot(2, 1, 2)
    y_pos = np.arange(len(predictions))
    scores = [float(pred['score'].strip('%')) for pred in predictions]
    labels = [f"{pred['label']} ({pred['score']})" for pred in predictions]
    
    plt.barh(y_pos, scores, align='center', alpha=0.8)
    plt.yticks(y_pos, labels)
    plt.xlabel('Confidence (%)')
    plt.title('Top 3 Predictions')
    plt.tight_layout()
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Image Classification using VGG16')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--no-display', action='store_true', help='Disable image display')
    args = parser.parse_args()
    
    try:
        # Load the pre-trained VGG16 model
        print("Loading VGG16 model (this may take a moment)...")
        model = VGG16(weights='imagenet')
        
        # Classify the image
        print(f"Classifying image: {args.image_path}")
        predictions = classify_image(model, args.image_path)
        
        # Display results
        print("\nTop 3 Predictions:")
        print("-" * 50)
        for pred in predictions:
            print(f"{pred['rank']}. {pred['label']}: {pred['score']}")
        
        # Display image and predictions if not disabled
        if not args.no_display:
            try:
                display_results(args.image_path, predictions)
            except Exception as e:
                print(f"\nNote: Could not display results: {str(e)}")
                print("Make sure you have a display available or use --no-display flag.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
