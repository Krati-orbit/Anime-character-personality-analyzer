# ==============================================================================
# MODEL CONFIGURATION & PREDICTION ENGINE (model.py)
# This file defines the 10-dimensional traits, the character prototypes, the
# questions schema, and computes similarity scores using Cosine Similarity.
# ==============================================================================

import numpy as np
import joblib
import os
import json

# ------------------------------------------------------------------------------
# THE 10 PERSONALITY DIMENSIONS (Feature Space)
# Every user profile and character is represented as a numerical vector:
# [Extraversion, Intellect, Discipline, Empathy, Determination, Aggression,
#  Optimism, Mysteriousness, Pride, Sacrifice] with scores ranging from 0.0 to 1.0.
# ------------------------------------------------------------------------------
TRAIT_LABELS = [
    "Extraversion",     # Loud, Energetic, Outgoing vs. Calm, Quiet
    "Intellect",        # Genius, Strategic, Calculative vs. Simple, Carefree
    "Discipline",       # Disciplined, Hardworking, Perfectionist vs. Reckless, Free-spirited
    "Empathy",          # Friendly, Loyal, Protective vs. Manipulative, Proud
    "Determination",    # Never gives up, Determined vs. Easygoing
    "Aggression",       # Battle-lover, Strong vs. Calm, Avoids fight
    "Optimism",         # Optimistic, Fun vs. Serious, Cynical
    "Mysteriousness",   # Mysterious, Stoic vs. Open, Loud
    "Pride",            # Proud, Arrogant, Justice-obsessed vs. Humble
    "Sacrifice"         # Sacrificing, Protective vs. Self-preserving
]

# ------------------------------------------------------------------------------
# CHARACTER PROTOTYPES (Centroids in the 10D space)
# These represent the 'ideal' trait scores for each counterpart.
# They act as the center of gravity for synthetic training data generation.
# Dynamically loaded from characters.json.
# ------------------------------------------------------------------------------
CHARACTER_PROTOTYPES = {}

def reload_prototypes():
    """Reloads character prototypes from characters.json in-place."""
    CHARACTER_PROTOTYPES.clear()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    CHARACTERS_FILE = os.path.join(current_dir, "characters.json")
    try:
        if os.path.exists(CHARACTERS_FILE):
            with open(CHARACTERS_FILE, "r", encoding="utf-8") as f:
                db = json.load(f)
                for slug, data in db.items():
                    if "vector" in data:
                        CHARACTER_PROTOTYPES[slug] = data["vector"]
    except Exception as e:
        print(f"Error loading prototypes in model.py: {e}")

# Initial load
reload_prototypes()

# ------------------------------------------------------------------------------
# QUIZ QUESTIONS SCHEMATICS
# Each of the 10 questions has 4 options.
# Each option is mapped to a 10D vector representing the trait influence.
# ------------------------------------------------------------------------------
QUESTIONS = [
    {
        "id": 1,
        "question": "How do you handle a big problem?",
        "options": [
            {
                "text": "Confront it directly with passion and energy, refusing to back down!",
                "vector": [0.9, 0.1, 0.2, 0.8, 1.0, 0.7, 0.9, 0.1, 0.4, 0.7] # Naruto/Luffy
            },
            {
                "text": "Analyze it silently, formulate a strategic plan, and execute it calmly.",
                "vector": [0.2, 1.0, 0.9, 0.4, 0.8, 0.3, 0.3, 0.8, 0.7, 0.6] # Itachi/Light
            },
            {
                "text": "Treat it as an exciting challenge to test your strength and grow stronger!",
                "vector": [0.8, 0.1, 0.5, 0.7, 0.9, 1.0, 0.9, 0.1, 0.5, 0.6] # Goku
            },
            {
                "text": "Deal with it calmly, focusing on efficiency and getting the job done perfectly.",
                "vector": [0.2, 0.7, 1.0, 0.6, 0.9, 0.8, 0.2, 0.6, 0.4, 0.8] # Levi
            }
        ]
    },
    {
        "id": 2,
        "question": "What is your role in a friend group?",
        "options": [
            {
                "text": "The energetic lifewire who keeps everyone laughing and motivated.",
                "vector": [1.0, 0.1, 0.2, 0.9, 0.9, 0.6, 1.0, 0.1, 0.4, 0.6] # Naruto/Luffy
            },
            {
                "text": "The quiet, mysterious guardian who protects them from the shadows.",
                "vector": [0.1, 0.8, 0.8, 0.9, 0.8, 0.3, 0.3, 1.0, 0.3, 1.0] # Itachi
            },
            {
                "text": "The simple, carefree one who is always ready for adventure and food.",
                "vector": [0.7, 0.1, 0.4, 0.8, 0.8, 0.9, 0.8, 0.1, 0.5, 0.6] # Goku
            },
            {
                "text": "The disciplined, blunt realist who keeps everyone in check and on track.",
                "vector": [0.2, 0.7, 1.0, 0.7, 0.8, 0.8, 0.2, 0.7, 0.3, 0.8] # Levi
            }
        ]
    },
    {
        "id": 3,
        "question": "How do you react when someone betrays you?",
        "options": [
            {
                "text": "Forgive them and try to win back their friendship, no matter what.",
                "vector": [0.9, 0.2, 0.4, 1.0, 1.0, 0.5, 0.9, 0.1, 0.3, 0.9] # Naruto
            },
            {
                "text": "Understand their motives, sacrifice your feelings, and protect the bigger picture.",
                "vector": [0.2, 0.9, 0.8, 0.8, 0.8, 0.4, 0.2, 0.9, 0.4, 1.0] # Itachi
            },
            {
                "text": "Fight them to teach them a lesson, but carry no long-term grudges.",
                "vector": [0.8, 0.1, 0.6, 0.8, 0.9, 0.9, 0.8, 0.1, 0.5, 0.7] # Goku
            },
            {
                "text": "Manipulate the situation to deliver your absolute justice.",
                "vector": [0.4, 1.0, 0.9, 0.1, 0.9, 0.5, 0.3, 0.8, 1.0, 0.1] # Light
            }
        ]
    },
    {
        "id": 4,
        "question": "What motivates you the most?",
        "options": [
            {
                "text": "The dream of achieving a legendary title and proving myself to everyone.",
                "vector": [0.9, 0.2, 0.3, 0.9, 1.0, 0.7, 0.9, 0.1, 0.6, 0.7] # Naruto/Luffy
            },
            {
                "text": "Protecting my loved ones and bringing peace, even if I must suffer in silence.",
                "vector": [0.1, 0.8, 0.9, 0.8, 0.9, 0.6, 0.2, 0.8, 0.3, 0.9] # Itachi/Levi
            },
            {
                "text": "Finding strong challenges and pushing my limits to become the strongest version of myself.",
                "vector": [0.8, 0.1, 0.6, 0.7, 0.9, 1.0, 0.9, 0.1, 0.6, 0.6] # Goku
            },
            {
                "text": "Creating a perfect, just world by purging those who act dishonorably.",
                "vector": [0.5, 1.0, 0.9, 0.1, 0.9, 0.4, 0.4, 0.8, 1.0, 0.1] # Light
            }
        ]
    },
    {
        "id": 5,
        "question": "How do you make important decisions?",
        "options": [
            {
                "text": "Trust my gut feelings and follow my heart, regardless of the rules or risks.",
                "vector": [0.9, 0.1, 0.2, 0.9, 1.0, 0.7, 0.9, 0.1, 0.4, 0.7] # Luffy/Naruto
            },
            {
                "text": "Calculate every possibility, weigh the strategic outcomes, and decide logically.",
                "vector": [0.3, 1.0, 0.9, 0.3, 0.8, 0.4, 0.3, 0.8, 0.8, 0.4] # Itachi/Light
            },
            {
                "text": "Choose the path that offers the most exciting challenge and personal growth.",
                "vector": [0.8, 0.1, 0.5, 0.8, 0.9, 0.9, 0.9, 0.1, 0.5, 0.6] # Goku
            },
            {
                "text": "Make the hard, practical choice based on duty and strict discipline.",
                "vector": [0.2, 0.7, 1.0, 0.6, 0.9, 0.8, 0.2, 0.7, 0.3, 0.9] # Levi
            }
        ]
    },
    {
        "id": 6,
        "question": "What is your fighting style? (metaphorically)",
        "options": [
            {
                "text": "Raw energy, loud shouts, and fighting with absolute passion and emotion.",
                "vector": [1.0, 0.2, 0.3, 0.9, 1.0, 0.8, 0.9, 0.1, 0.5, 0.8] # Naruto
            },
            {
                "text": "Strategic, deceptive, and predicting the opponent's moves steps ahead.",
                "vector": [0.2, 1.0, 0.9, 0.3, 0.8, 0.3, 0.3, 0.9, 0.8, 0.6] # Itachi/Light
            },
            {
                "text": "Martial arts mastery, focusing on power, speed, and purely enjoying the battle.",
                "vector": [0.8, 0.1, 0.6, 0.7, 0.9, 1.0, 0.9, 0.1, 0.6, 0.7] # Goku
            },
            {
                "text": "Swift, disciplined precision, leaving zero margin for error.",
                "vector": [0.2, 0.7, 1.0, 0.6, 0.9, 0.9, 0.2, 0.7, 0.4, 0.9] # Levi
            }
        ]
    },
    {
        "id": 7,
        "question": "What do people misunderstand about you?",
        "options": [
            {
                "text": "They think I'm just loud and silly, but I carry deep pain and care for everyone.",
                "vector": [0.9, 0.3, 0.4, 0.9, 1.0, 0.6, 0.9, 0.3, 0.4, 0.8] # Naruto
            },
            {
                "text": "They think I am cold and heartless, but I am actually deeply protective and caring.",
                "vector": [0.1, 0.8, 0.9, 0.8, 0.9, 0.7, 0.2, 0.8, 0.3, 0.9] # Levi/Itachi
            },
            {
                "text": "They think I'm simple and naive, but I possess an iron will that cannot be broken.",
                "vector": [0.8, 0.1, 0.4, 0.9, 1.0, 0.9, 0.9, 0.1, 0.5, 0.7] # Goku/Luffy
            },
            {
                "text": "They think I'm just a normal, model citizen, but I hold supreme ambition.",
                "vector": [0.5, 0.9, 0.9, 0.2, 0.9, 0.5, 0.5, 0.7, 1.0, 0.2] # Light
            }
        ]
    },
    {
        "id": 8,
        "question": "Your biggest strength?",
        "options": [
            {
                "text": "My unwavering determination—I never, ever give up, no matter the odds.",
                "vector": [0.9, 0.1, 0.2, 0.9, 1.0, 0.8, 1.0, 0.1, 0.5, 0.8] # Naruto/Luffy
            },
            {
                "text": "My mind—high intelligence, strategic plotting, and analytical depth.",
                "vector": [0.3, 1.0, 0.9, 0.4, 0.8, 0.4, 0.3, 0.9, 0.8, 0.6] # Itachi/Light
            },
            {
                "text": "My ability to push physical boundaries and adapt in the heat of battle.",
                "vector": [0.8, 0.1, 0.6, 0.8, 0.9, 1.0, 0.9, 0.1, 0.6, 0.7] # Goku
            },
            {
                "text": "My flawless discipline, execution speed, and absolute focus.",
                "vector": [0.2, 0.7, 1.0, 0.7, 0.9, 0.9, 0.2, 0.7, 0.4, 0.9] # Levi
            }
        ]
    },
    {
        "id": 9,
        "question": "How do you spend your free time?",
        "options": [
            {
                "text": "Hanging out with close friends, sharing a hearty meal, and laughing.",
                "vector": [0.9, 0.1, 0.3, 0.9, 0.9, 0.7, 0.9, 0.1, 0.5, 0.7] # Naruto/Luffy/Goku
            },
            {
                "text": "Reflecting in peaceful solitude, reading, or planning future goals.",
                "vector": [0.2, 1.0, 0.9, 0.3, 0.8, 0.3, 0.3, 0.9, 0.8, 0.5] # Itachi/Light
            },
            {
                "text": "Training hard to sharpen my skills, or exploring challenging new environments.",
                "vector": [0.6, 0.3, 0.8, 0.7, 0.9, 0.9, 0.6, 0.4, 0.5, 0.8] # Goku/Levi
            },
            {
                "text": "Cleaning and organizing my space meticulously so that everything is in its place.",
                "vector": [0.2, 0.6, 1.0, 0.6, 0.8, 0.7, 0.2, 0.6, 0.3, 0.7] # Levi cleaning
            }
        ]
    },
    {
        "id": 10,
        "question": "What is your life goal?",
        "options": [
            {
                "text": "To live with complete freedom, embark on adventures, and protect my crew.",
                "vector": [0.9, 0.1, 0.3, 0.9, 1.0, 0.9, 1.0, 0.1, 0.5, 0.7] # Luffy/Goku
            },
            {
                "text": "To lead, protect my community, and ultimately gain the respect of everyone.",
                "vector": [1.0, 0.3, 0.4, 0.9, 1.0, 0.7, 1.0, 0.2, 0.5, 0.8] # Naruto
            },
            {
                "text": "To ensure peace and stability, even if it requires carrying a heavy cross.",
                "vector": [0.1, 0.8, 0.9, 0.8, 0.9, 0.6, 0.2, 0.8, 0.4, 1.0] # Itachi/Levi
            },
            {
                "text": "To rewrite the system, enforce absolute order, and eliminate corruption.",
                "vector": [0.5, 1.0, 0.9, 0.1, 0.9, 0.5, 0.4, 0.8, 1.0, 0.1] # Light
            }
        ]
    }
]

# ------------------------------------------------------------------------------
# FUNCTION: get_user_vector
# Averages the 10D vectors of all selected quiz answers.
# ------------------------------------------------------------------------------
def get_user_vector(answer_indices):
    """
    Given a list of 10 answer indices (0-3), calculate the average 10-dimensional trait vector.
    """
    if len(answer_indices) != 10:
        raise ValueError("Exactly 10 answers must be provided.")
    
    vectors = []
    # Loop through each question index and the selected option index
    for q_idx, opt_idx in enumerate(answer_indices):
        if not (0 <= opt_idx <= 3):
            raise ValueError(f"Invalid option index {opt_idx} at question {q_idx + 1}")
        # Fetch the option's 10D vector
        vectors.append(QUESTIONS[q_idx]["options"][opt_idx]["vector"])
    
    # Return the mean average vector along the traits column axis (axis 0)
    # This results in a final user vector representing their composite score in all 10 traits.
    return np.mean(vectors, axis=0)


# ------------------------------------------------------------------------------
# FUNCTION: predict_character
# Queries the trained KNN model to classify the user's vector,
# then uses Cosine Similarity to calculate the compatibility percentage.
# ------------------------------------------------------------------------------
def predict_character(user_vector):
    """
    Predict the matched character name and match percentage using similarity metrics.
    Loads the trained KNN model to classify, then calculates cosine similarity for match percentage.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "knn_model.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained KNN model not found. Please run train.py first.")
    
    # Load serialized model details
    model_data = joblib.load(model_path)
    knn_model = model_data["model"]
    class_names = model_data["class_names"]
    
    # Reshape 1D vector to 2D shape (1, 10) for Scikit-Learn input compatibility
    user_vector_2d = user_vector.reshape(1, -1)
    
    # Query KNN to find the nearest matched prototype index
    prediction_idx = knn_model.predict(user_vector_2d)[0]
    predicted_slug = class_names[prediction_idx]
    
    # --- COSINE SIMILARITY MATH ---
    # Formula: similarity = (A . B) / (||A|| * ||B||)
    # Cosine Similarity is highly descriptive because it measures the angle/direction
    # of the personality profile instead of just the absolute scale size.
    proto_vector = np.array(CHARACTER_PROTOTYPES[predicted_slug])
    
    dot_product = np.dot(user_vector, proto_vector)      # A . B
    norm_user = np.linalg.norm(user_vector)              # ||A|| (magnitude)
    norm_proto = np.linalg.norm(proto_vector)            # ||B|| (magnitude)
    
    if norm_user == 0 or norm_proto == 0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = dot_product / (norm_user * norm_proto)
        
    # Scale decimal to integer percentage
    match_percentage = int(cosine_similarity * 100)
    
    # Clip to standard display boundaries (e.g. 50% to 99% for a better feel)
    match_percentage = max(50, min(99, match_percentage))
    
    return predicted_slug, match_percentage


# ------------------------------------------------------------------------------
# FUNCTION: get_rival_character
# Evaluates which character has the lowest Cosine Similarity (widest angle)
# relative to the user's vector to find their personality rival.
# ------------------------------------------------------------------------------
def get_rival_character(user_vector):
    """
    Find the character with the lowest cosine similarity to the user's vector.
    Returns: (rival_slug, compatibility_percentage)
    """
    lowest_similarity = 2.0
    rival_slug = None
    
    # Iterate through all prototypes to find the minimum similarity coefficient
    for slug, proto_list in CHARACTER_PROTOTYPES.items():
        proto_vector = np.array(proto_list)
        dot_product = np.dot(user_vector, proto_vector)
        norm_user = np.linalg.norm(user_vector)
        norm_proto = np.linalg.norm(proto_vector)
        
        if norm_user == 0 or norm_proto == 0:
            sim = 0.0
        else:
            sim = dot_product / (norm_user * norm_proto)
            
        if sim < lowest_similarity:
            lowest_similarity = sim
            rival_slug = slug
            
    # Scale decimal to integer percentage (representing low similarity)
    rival_match_score = int(lowest_similarity * 100)
    # Clip rival match percentage to 20-50% boundaries for rendering
    rival_match_score = max(20, min(50, rival_match_score))
    
    return rival_slug, rival_match_score

