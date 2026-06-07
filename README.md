---
title: Anime Character Personality Analyzer
emoji: 🧪
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Anime Character Personality Analyzer 🧪✨

A premium, modern web application that analyzes your personality through a 10-question quiz and matches you with an iconic anime character. Under the hood, it projects your choices into a 10-dimensional personality trait vector and classifies it using a **K-Nearest Neighbors (KNN) Machine Learning model (n_neighbors=3)**.Here's the link of my deployed project " https://huggingface.co/spaces/krati-orbit/Anime-character-personality-analyzer".

---

## 🚀 Tech Stack

- **Backend**: Python + Flask
- **Machine Learning**: Scikit-Learn (KNN Classifier) + Joblib (for saving/loading)
- **Frontend**: HTML5 + CSS3 (No framework, pure premium styles) + Vanilla JavaScript
- **Data**: JSON Database (`characters.json`)

---

## 📂 Project Structure

```text
anime-personality-analyzer/
├── app.py                # Flask Web Server
├── model.py              # Quiz definition, vector conversion & prediction
├── train.py              # ML model trainer (generates synthetic dataset & trains KNN)
├── characters.json       # Character database (info, quotes, accent colors, traits)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation (this file)
├── templates/
│   ├── index.html        # Landing screen & Quiz layout
│   └── result.html       # Dynamic result presentation with accent customization
└── static/
    ├── style.css         # Dark themed CSS with premium visual elements & transitions
    ├── script.js         # Frontend quiz engine, chakra spinner, and API calls
    └── images/
        └── .gitkeep      # Folder for character photos
```

---

## ⚡ How it Works (The Machine Learning Magic)

1. **Dimensionality**: The application maps your answers to **10 personality dimensions**:
   - *Extraversion, Intellect, Discipline, Empathy, Determination, Aggression, Optimism, Mysteriousness, Pride, Sacrifice*.
2. **Synthetic Training Data**: `train.py` establishes prototype vectors for each character (e.g. Naruto has high Extraversion and Determination, Itachi has high Intellect and Sacrifice). It generates a synthetic dataset of **1,200 records** by applying Gaussian noise around the prototypes.
3. **KNN Classification**: A `KNeighborsClassifier` with $k=3$ fits this synthetic data. When you submit your answers, the model finds the 3 nearest neighbors in vector space.
4. **Compatibility Score**: To compute your specific match percentage, the application calculates the **Cosine Similarity** between your averaged vector and the matched character's prototype vector.

---

## 🛠️ Setup & Local Execution

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. In your terminal, run:
```bash
pip install -r requirements.txt
```

### 2. Train the ML Model
Generate the training dataset and save the trained classifier:
```bash
python train.py
```
*This will create the `knn_model.joblib` file in the root directory.*

### 3. Run the Web App
Start the local development server:
```bash
python app.py
```
*Open `http://127.0.0.1:5000` in your web browser to take the quiz.*

---

## 🖼️ Adding Character Images

The application comes with a dynamic image loader. If you do not have character images ready, the result page displays a gorgeous glassmorphism placeholder representing the character's emoji and accent color.

To add images:
1. Save your image files in the `static/images/` folder.
2. Ensure they are named **exactly** as follows (as linked in `characters.json`):
   - `naruto.png` (or `.jpg`)
   - `itachi.png`
   - `goku.png`
   - `light.png`
   - `luffy.png`
   - `levi.png`
3. If you use a different format (like `.jpg` or `.webp`), open `characters.json` and update the `"image"` path for the respective character.
