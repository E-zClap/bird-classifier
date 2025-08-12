#!/usr/bin/env python3
"""
Predict bird species from an image using the trained classifier.
"""

import argparse
import matplotlib.pyplot as plt
from src.inference import load_model_and_predict


def main():
    parser = argparse.ArgumentParser(description='Predict bird species from an image')
    parser.add_argument('image_path', type=str,
                        help='Path to the input image')
    parser.add_argument('--model-path', type=str, default='models/bird_species_classifier.pt',
                        help='Path to trained model')
    parser.add_argument('--show-plot', action='store_true',
                        help='Show visualization of prediction')
    
    args = parser.parse_args()
    
    # Make prediction
    try:
        predicted_class, confidence = load_model_and_predict(args.image_path, args.model_path)
        print(f"Predicted bird species: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error making prediction: {e}")
        return
    
    if args.show_plot:
        plt.show()


if __name__ == '__main__':
    main()
