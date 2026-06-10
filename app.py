# ==============================================================================
# FLASK WEB SERVER CONTROLLER (app.py)
# This file handles HTTP requests, routes, session management, and log storage.
# ==============================================================================

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json
import os
import logging
import csv
from datetime import datetime
from functools import wraps
# Import model helpers for converting choices, predicting matches, and comparing rivals
from model import get_user_vector, predict_character, get_rival_character, CHARACTER_PROTOTYPES

# Configure application logging (displays server logs in console)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask Application
app = Flask(__name__)
# Secret key used for signing cookies/session tokens securely
app.secret_key = "anime_personality_secret_key_987654"
# Hardcoded admin credentials for system dashboard lockscreen
ADMIN_PASSWORD = "admin123"
ADMIN_EMAIL = "krati2510@gmail.com"
# CSV file path where all participant test results will be appended and stored
CSV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users_log.csv")

# Initialize and load characters database from JSON file
CHARACTERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "characters.json")

def load_characters_db():
    """Loads character profiles from characters.json."""
    try:
        if os.path.exists(CHARACTERS_FILE):
            with open(CHARACTERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading characters.json: {e}")
    return {}

def save_characters_db(db):
    """Saves character profiles to characters.json and reloads prototype vectors."""
    try:
        with open(CHARACTERS_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
        global characters_db
        characters_db = db
        # Force model.py to reload the prototype mappings
        from model import reload_prototypes
        reload_prototypes()
        logger.info("Successfully updated characters.json and reloaded prototypes.")
        return True
    except Exception as e:
        logger.error(f"Error saving characters.json: {e}")
        return False

# Load global cache on startup
characters_db = load_characters_db()

# ------------------------------------------------------------------------------
# ROUTE: Homepage (/)
# Renders the main quiz taking layout.
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    """Renders the quiz landing/taking page (index.html)."""
    global characters_db
    characters_db = load_characters_db()
    return render_template("index.html", characters=characters_db)

# ------------------------------------------------------------------------------
# ROUTE: Quiz Prediction AJAX Endpoint (/predict)
# Receives user choices, calculates personality vectors, queries the KNN model,
# appends the result log to CSV, and returns JSON metadata.
# ------------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles the AJAX submission of quiz answer indices.
    Expects JSON input: { "answers": [0, 2, 1, 3, ...], "name": "...", "age": "...", "gender": "..." }
    """
    try:
        # Parse JSON payload from POST request
        data = request.get_json()
        if not data or "answers" not in data:
            return jsonify({"error": "Invalid request. No answers provided."}), 400
        
        answers = data["answers"]
        # Ensure exactly 10 questions are answered
        if not isinstance(answers, list) or len(answers) != 10:
            return jsonify({"error": "Exactly 10 answers are required."}), 400
        
        # Extract optional user profile fields (default to placeholders if omitted)
        name = data.get("name", "User")
        age = data.get("age", "N/A")
        gender = data.get("gender", "N/A")
        
        # Parse indices to integers safely
        answer_indices = [int(x) for x in answers]
        
        # 1. Convert answers to 10-dimensional trait vector
        # (This averages the vectors of the selected options in model.py)
        user_vector = get_user_vector(answer_indices)
        
        # 2. Predict matched character and compatibility percentage using the ML model
        # (Queries local knn_model.joblib classifier and performs Cosine Similarity math)
        predicted_slug, match_percentage = predict_character(user_vector)
        
        # 3. Find the lowest similarity character as the 'anime rival' counterpart
        rival_slug, rival_score = get_rival_character(user_vector)
        
        logger.info(f"Quiz completed. Matched user with: {predicted_slug} ({match_percentage}%)")
        
        # 4. Append the user's test profile and results to the local CSV log file
        try:
            file_exists = os.path.exists(CSV_FILE_PATH)
            with open(CSV_FILE_PATH, mode="a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                # If file doesn't exist, create it and write column headers first
                if not file_exists:
                    writer.writerow(["timestamp", "name", "age", "gender", "matched_character", "compatibility_score"])
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Remove commas from the participant's name to avoid breaking CSV columns
                clean_name = str(name).replace(",", " ")
                writer.writerow([timestamp, clean_name, age, gender, predicted_slug.capitalize(), f"{match_percentage}%"])
        except Exception as csv_err:
            logger.error(f"Error logging quiz result to CSV: {csv_err}")
            
        # Return matched data back to JavaScript AJAX handler
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

# ------------------------------------------------------------------------------
# ROUTE: Result Page (/result)
# Renders the results dashboard based on query parameters.
# ------------------------------------------------------------------------------
@app.route("/result")
def result():
    """
    Renders the result page for the matched character.
    Query parameters: char=<slug>&score=<int>&rival=<slug>&rival_score=<int>
    """
    global characters_db
    characters_db = load_characters_db()
    
    # Extract query params from redirection URL
    char_slug = request.args.get("char", "").lower()
    score_str = request.args.get("score", "0")
    rival_slug = request.args.get("rival", "").lower()
    rival_score_str = request.args.get("rival_score", "0")
    
    # If the character slug doesn't exist, redirect back to home page
    if not char_slug or char_slug not in characters_db:
        logger.warning(f"Access to result page with invalid character slug: '{char_slug}'")
        return redirect(url_for("index"))
    
    # Sanitize compatibility score input
    try:
        score = int(score_str)
        if not (0 <= score <= 100):
            score = 75  # Default fallback
    except ValueError:
        score = 75  # Default fallback
        
    # Sanitize rival compatibility score input
    try:
        rival_score = int(rival_score_str)
        if not (0 <= rival_score <= 100):
            rival_score = 35
    except ValueError:
        rival_score = 35

    character_info = characters_db[char_slug]
    
    # Extract hex color and convert to RGB values (e.g. "#ff6600" -> "255, 102, 0")
    # This is passed to the frontend to build rgba() overlays dynamically in CSS
    color_hex = character_info.get("color", "#ffffff").lstrip('#')
    try:
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
        rgb_color = f"{r}, {g}, {b}"
    except Exception:
        rgb_color = "255, 255, 255"  # Fallback to white if hex parse fails
    
    # Get matched character prototype vector for rendering comparison chart
    char_vector = CHARACTER_PROTOTYPES.get(char_slug, [0]*10)
    
    # Get rival details (calculate dynamically if missing from query params)
    if not rival_slug or rival_slug not in characters_db:
        import numpy as np
        rival_slug, rival_score = get_rival_character(np.array(char_vector))
        
    rival_character = characters_db[rival_slug]
        
    # Render final page passing in template context
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


# ------------------------------------------------------------------------------
# ROUTE: Image Proxy (/image_proxy)
# Downloads remote Anilist image data server-side and routes it back to the client.
# This prevents CORS errors when exporting canvas card screenshots with html2canvas.
# ------------------------------------------------------------------------------
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
        # Spoof a standard browser user agent and referer to bypass hotlink blocks on AniList CDN
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
            
        # Return binary image response with matching Content-Type and CORS header (*)
        res = app.make_response(img_data)
        res.headers.set("Content-Type", content_type)
        res.headers.set("Access-Control-Allow-Origin", "*")
        return res
    except Exception as e:
        logger.error(f"Error proxying image {url}: {e}")
        return "Error loading image", 500


# ------------------------------------------------------------------------------
# DECORATOR: admin_required
# Wraps view functions to block non-authenticated access.
# ------------------------------------------------------------------------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Redirect to login screen if administrator is not logged in
        if not session.get("logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function

# ------------------------------------------------------------------------------
# ROUTE: Admin Login (/admin/login)
# Renders password lockscreen page and evaluates credentials.
# ------------------------------------------------------------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Renders the lock screen and handles admin authentication."""
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password")
        
        # Compare credentials
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            logger.info("Admin logged in successfully.")
            # Redirect to admin selection portal instead of dashboard logs
            return redirect(url_for("admin_portal"))
        else:
            error = "Invalid email or password. Access Denied."
            logger.warning("Failed admin login attempt.")
            
    return render_template("admin_login.html", error=error)

# ------------------------------------------------------------------------------
# ROUTE: Admin Options Portal (/admin/portal)
# Selection hub showing two pathways (take quiz or view dashboard logs).
# ------------------------------------------------------------------------------
@app.route("/admin/portal")
@admin_required
def admin_portal():
    """Renders the admin control hub/options page."""
    return render_template("admin_portal.html")

# ------------------------------------------------------------------------------
# ROUTE: Admin Dashboard (/admin)
# Displays all test results recorded in the CSV log.
# ------------------------------------------------------------------------------
@app.route("/admin")
@admin_required
def admin_dashboard():
    """Renders the admin panel with a list of all user test submissions."""
    users_list = []
    # Read telemetry logs from CSV file
    if os.path.exists(CSV_FILE_PATH):
        try:
            with open(CSV_FILE_PATH, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    users_list.append(row)
        except Exception as e:
            logger.error(f"Error reading CSV logs: {e}")
            
    # Reverse list to show the most recent submissions at the top
    users_list.reverse()
    
    # Load latest characters database
    global characters_db
    characters_db = load_characters_db()
    
    return render_template(
        "admin_dashboard.html",
        users=users_list,
        total_count=len(users_list),
        characters=characters_db
    )

# ------------------------------------------------------------------------------
# ROUTE: Save Character Details (/admin/characters/save)
# Adds a new character or updates an existing one, reloading the prototype mappings.
# ------------------------------------------------------------------------------
@app.route("/admin/characters/save", methods=["POST"])
@admin_required
def admin_save_character():
    """Handles adding or editing an anime character."""
    try:
        data = request.get_json()
        if not data or "slug" not in data or "name" not in data:
            return jsonify({"success": False, "error": "Invalid request parameters."}), 400
        
        slug = data["slug"].strip().lower()
        if not slug.isalnum():
            return jsonify({"success": False, "error": "Slug must be alphanumeric (no spaces or special chars)."}), 400
            
        name = data["name"].strip()
        anime = data.get("anime", "").strip()
        emoji = data.get("emoji", "").strip()
        color = data.get("color", "#ffffff").strip()
        quote = data.get("quote", "").strip()
        image = data.get("image", "").strip()
        
        # Parse traits (from array of strings or comma-separated string)
        traits = data.get("traits", [])
        if isinstance(traits, str):
            traits = [t.strip() for t in traits.split(",") if t.strip()]
            
        description = data.get("description", "").strip()
        
        # Parse and validate the 10D vector
        vector = data.get("vector", [])
        if not isinstance(vector, list) or len(vector) != 10:
            return jsonify({"success": False, "error": "Personality vector must contain exactly 10 values."}), 400
            
        vector = [max(0.0, min(1.0, float(v))) for v in vector]
        
        # Load, modify, and save database
        db = load_characters_db()
        db[slug] = {
            "name": name,
            "anime": anime,
            "emoji": emoji,
            "color": color,
            "quote": quote,
            "image": image,
            "traits": traits,
            "description": description,
            "vector": vector
        }
        
        if save_characters_db(db):
            return jsonify({"success": True, "message": f"Character '{name}' saved successfully."})
        else:
            return jsonify({"success": False, "error": "Failed to write character data to database."}), 500
            
    except Exception as e:
        logger.error(f"Error saving character: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ------------------------------------------------------------------------------
# ROUTE: Delete Character (/admin/characters/delete/<slug>)
# Deletes a character from the database and updates the prototypes.
# ------------------------------------------------------------------------------
@app.route("/admin/characters/delete/<slug>", methods=["POST"])
@admin_required
def admin_delete_character(slug):
    """Handles deleting a character from the database."""
    try:
        slug = slug.strip().lower()
        db = load_characters_db()
        if slug not in db:
            return jsonify({"success": False, "error": f"Character '{slug}' not found."}), 404
            
        name = db[slug].get("name", slug)
        del db[slug]
        
        if save_characters_db(db):
            return jsonify({"success": True, "message": f"Character '{name}' deleted successfully."})
        else:
            return jsonify({"success": False, "error": "Failed to write database after deletion."}), 500
            
    except Exception as e:
        logger.error(f"Error deleting character: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ------------------------------------------------------------------------------
# ROUTE: Retrain ML Model (/admin/retrain)
# Calls train_model from train.py to regenerate dataset and update KNN.
# ------------------------------------------------------------------------------
@app.route("/admin/retrain", methods=["POST"])
@admin_required
def admin_retrain_model():
    """Triggers retraining of the KNN classifier model."""
    try:
        from train import train_model
        # Execute training logic synchronously (fast since it generates synthetic data)
        train_model()
        return jsonify({"success": True, "message": "ML Classifier model retrained successfully."})
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ------------------------------------------------------------------------------
# ROUTE: Admin Logout (/admin/logout)
# Clears active administrator session cookies.
# ------------------------------------------------------------------------------
@app.route("/admin/logout")
def admin_logout():
    """Clears the admin login session."""
    session.pop("logged_in", None)
    logger.info("Admin logged out.")
    return redirect(url_for("index"))


# ------------------------------------------------------------------------------
# GLOBAL ERROR ROUTE: 404 Not Found
# Catch-all page redirect back to homepage.
# ------------------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("index"))

# ------------------------------------------------------------------------------
# MAIN APPLICATION STARTER
# Configures server ports and host mappings for Docker / Hugging Face Spaces.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Pull dynamic port assigned by Docker host (defaults to 7860 on Hugging Face Spaces)
    port = int(os.environ.get("PORT", 7860))
    # Toggle debug mode locally (when FLASK_ENV=development is set)
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(debug=debug_mode, host="0.0.0.0", port=port)

