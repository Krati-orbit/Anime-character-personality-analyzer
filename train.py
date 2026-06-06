import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
from model import CHARACTER_PROTOTYPES

def train_model():
    print("Generating synthetic dataset for character personality training...")
    
    # Standard labels corresponding to characters
    char_slugs = list(CHARACTER_PROTOTYPES.keys())
    
    X = []
    y = []
    
    # We will generate 200 samples per character to ensure robust KNN boundary mapping
    samples_per_character = 200
    np.random.seed(42) # Set seed for reproducibility
    
    for idx, slug in enumerate(char_slugs):
        prototype = np.array(CHARACTER_PROTOTYPES[slug])
        
        for _ in range(samples_per_character):
            # Add Gaussian noise to the prototype vector (std = 0.12)
            noise = np.random.normal(0, 0.12, size=len(prototype))
            synthetic_vector = prototype + noise
            
            # Clip values to remain within [0.0, 1.0] bounds
            synthetic_vector = np.clip(synthetic_vector, 0.0, 1.0)
            
            X.append(synthetic_vector)
            y.append(idx)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total training samples generated: {len(X)} across {len(char_slugs)} characters.")
    
    # Initialize KNN Classifier with n_neighbors = 3 (as specified)
    knn = KNeighborsClassifier(n_neighbors=3)
    
    # Train the model
    knn.fit(X, y)
    print("KNN Classifier model training complete.")
    
    # Self-evaluate training accuracy
    predictions = knn.predict(X)
    accuracy = np.mean(predictions == y) * 100
    print(f"Training accuracy: {accuracy:.2f}% (should be near 100% due to prototype separation)")
    
    # Save the model and labels list
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
