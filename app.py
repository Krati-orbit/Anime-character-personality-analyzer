from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
import logging
from model import get_user_vector, predict_character

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load character database
CHARACTERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "characters.json")
try:
    with open(CHARACTERS_FILE, "r", encoding="utf-8") as f:
        characters_db = json.load(f)
except Exception as e:
    logger.error(f"Error loading characters.json: {e}")
    characters_db = {}

@app.route("/")
def index():
    """Renders the quiz landing/taking page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles the AJAX submission of quiz answer indices.
    Expects JSON input: { "answers": [0, 2, 1, 3, ...] } (10 integer elements)
    """
    try:
        data = request.get_json()
        if not data or "answers" not in data:
            return jsonify({"error": "Invalid request. No answers provided."}), 400
        
        answers = data["answers"]
        if not isinstance(answers, list) or len(answers) != 10:
            return jsonify({"error": "Exactly 10 answers are required."}), 400
        
        # Parse indices to integers
        answer_indices = [int(x) for x in answers]
        
        # Convert answers to trait vector
        user_vector = get_user_vector(answer_indices)
        
        # Predict character and match percentage using ML model
        predicted_slug, match_percentage = predict_character(user_vector)
        
        logger.info(f"Quiz completed. Matched user with: {predicted_slug} ({match_percentage}%)")
        
        return jsonify({
            "success": True,
            "character": predicted_slug,
            "score": match_percentage
        })
        
    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except FileNotFoundError as fnfe:
        logger.error(f"Model file error: {fnfe}")
        return jsonify({"error": "The personality classification model is currently offline. Please try again later."}), 500
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}")
        return jsonify({"error": "An unexpected error occurred while processing your answers."}), 500

@app.route("/result")
def result():
    """
    Renders the result page for the matched character.
    Query parameters: char=<slug>&score=<int>
    """
    char_slug = request.args.get("char", "").lower()
    score_str = request.args.get("score", "0")
    
    if not char_slug or char_slug not in characters_db:
        logger.warning(f"Access to result page with invalid character slug: '{char_slug}'")
        return redirect(url_for("index"))
    
    try:
        score = int(score_str)
        if not (0 <= score <= 100):
            score = 75  # Default fallback
    except ValueError:
        score = 75  # Default fallback
        
    character_info = characters_db[char_slug]
    
    # Extract hex color and convert to RGB values (e.g. "#ff6600" -> "255, 102, 0")
    color_hex = character_info.get("color", "#ffffff").lstrip('#')
    try:
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
        rgb_color = f"{r}, {g}, {b}"
    except Exception:
        rgb_color = "255, 255, 255"  # Fallback to white
    
    return render_template(
        "result.html",
        character=character_info,
        score=score,
        slug=char_slug,
        rgb_color=rgb_color
    )


@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5080)

