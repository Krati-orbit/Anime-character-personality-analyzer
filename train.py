# ==============================================================================
# KNN CLASSIFIER MODEL TRAINER (train.py)
# This script generates a synthetic personality dataset based on the prototype
# vectors defined in model.py and fits a K-Nearest Neighbors Classifier.
# ==============================================================================

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
from model import CHARACTER_PROTOTYPES

def train_model():
    print("Generating synthetic dataset for character personality training...")
    
    # Extract list of character slugs (keys of CHARACTER_PROTOTYPES)
    # e.g., ['naruto', 'itachi', 'goku', 'light', 'luffy', 'levi', 'l', 'eren', 'tanjiro']
    char_slugs = list(CHARACTER_PROTOTYPES.keys())
    
    X = []  # Feature vectors list (10D personality score lists)
    y = []  # Label indices list (corresponds to indexes in char_slugs)
    
    # Number of synthetic samples to synthesize per character prototype.
    # Generating 200 samples ensures a rich classification volume for KNN boundary definitions.
    samples_per_character = 200
    np.random.seed(42)  # Fixed seed guarantees identical datasets across runs
    
    # Loop through all characters to synthesize variations
    for idx, slug in enumerate(char_slugs):
        prototype = np.array(CHARACTER_PROTOTYPES[slug])
        
        for _ in range(samples_per_character):
            # Generate Gaussian noise (Normal distribution) with standard deviation 0.12.
            # This simulates realistic human responses that deviate slightly from the ideal archetype.
            noise = np.random.normal(0, 0.12, size=len(prototype))
            synthetic_vector = prototype + noise
            
            # Clip values to remain strictly within [0.0, 1.0] limits
            synthetic_vector = np.clip(synthetic_vector, 0.0, 1.0)
            
            X.append(synthetic_vector)
            y.append(idx)
            
    # Convert lists to NumPy arrays (format expected by Scikit-Learn)
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total training samples generated: {len(X)} across {len(char_slugs)} characters.")
    
    # Initialize K-Nearest Neighbors Classifier.
    # n_neighbors=3 means that when a user takes the quiz, the model will find
    # the 3 closest synthetic samples in 10D space and assign the majority class.
    knn = KNeighborsClassifier(n_neighbors=3)
    
    # Train/Fit the model using the synthetic dataset
    knn.fit(X, y)
    print("KNN Classifier model training complete.")
    
    # Calculate training accuracy score to confirm successful prototype boundaries
    predictions = knn.predict(X)
    accuracy = np.mean(predictions == y) * 100
    print(f"Training accuracy: {accuracy:.2f}% (should be near 100% due to prototype separation)")
    
    # Save the trained model object and label maps to file
    model_data = {
        "model": knn,
        "class_names": char_slugs
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "knn_model.joblib")
    joblib.dump(model_data, save_path)
    print(f"Model successfully saved to: {save_path}")

if __name__ == "__main__":
    train_model()
