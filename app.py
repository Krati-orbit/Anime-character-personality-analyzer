from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
import logging
from model import get_user_vector, predict_character, get_rival_character, CHARACTER_PROTOTYPES

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
        rival_slug, rival_score = get_rival_character(user_vector)
        
        logger.info(f"Quiz completed. Matched user with: {predicted_slug} ({match_percentage}%)")
        
        return jsonify({
            "success": True,
            "character": predicted_slug,
            "score": match_percentage,
            "rival": rival_slug,
            "rival_score": rival_score,
            "user_vector": user_vector.tolist() if hasattr(user_vector, 'tolist') else list(user_vector)
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
    Query parameters: char=<slug>&score=<int>&rival=<slug>&rival_score=<int>
    """
    char_slug = request.args.get("char", "").lower()
    score_str = request.args.get("score", "0")
    rival_slug = request.args.get("rival", "").lower()
    rival_score_str = request.args.get("rival_score", "0")
    
    if not char_slug or char_slug not in characters_db:
        logger.warning(f"Access to result page with invalid character slug: '{char_slug}'")
        return redirect(url_for("index"))
    
    try:
        score = int(score_str)
        if not (0 <= score <= 100):
            score = 75  # Default fallback
    except ValueError:
        score = 75  # Default fallback
        
    try:
        rival_score = int(rival_score_str)
        if not (0 <= rival_score <= 100):
            rival_score = 35
    except ValueError:
        rival_score = 35

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
    
    # Get matched character prototype vector
    char_vector = CHARACTER_PROTOTYPES.get(char_slug, [0]*10)
    
    # Get rival details (calculate dynamically if missing from query params)
    if not rival_slug or rival_slug not in characters_db:
        import numpy as np
        rival_slug, rival_score = get_rival_character(np.array(char_vector))
        
    rival_character = characters_db[rival_slug]
        
    return render_template(
        "result.html",
        character=character_info,
        score=score,
        slug=char_slug,
        rgb_color=rgb_color,
        char_vector=char_vector,
        rival=rival_character,
        rival_score=rival_score,
        rival_slug=rival_slug
    )


@app.route("/image_proxy")
def image_proxy():
    """
    Proxies character image requests server-side to bypass CORS and hotlink protections.
    """
    url = request.args.get("url", "")
    if not url:
        return "Missing URL", 400
        
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer": "https://anilist.co/"
            }
        )
        with urllib.request.urlopen(req) as response:
            img_data = response.read()
            content_type = response.headers.get("Content-Type", "image/jpeg")
            
        res = app.make_response(img_data)
        res.headers.set("Content-Type", content_type)
        res.headers.set("Access-Control-Allow-Origin", "*")
        return res
    except Exception as e:
        logger.error(f"Error proxying image {url}: {e}")
        return "Error loading image", 500



@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5080)

