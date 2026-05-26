# ⚽ FIFA World Cup 2026 — Match Outcome Predictor

---

## 📌 Project Overview

This project is a **live AI-powered web application** that predicts FIFA World Cup 2026 match outcomes using machine learning.

Built on **49,000+ historical international fixtures** and **ELO rating systems**, the model analyzes team strengths, head-to-head records, and performance trends to predict winners across all tournament stages.

The project demonstrates a complete end-to-end ML workflow:

- Data collection & cleaning
- Feature engineering (ELO ratings, team stats, Dixon-Coles model)
- Model training with XGBoost
- Flask web app development
- Cloud deployment on Render

---

## 🌐 Live Demo

👉 **[https://fifa-world-cup-2k26-predictor.onrender.com](https://fifa-world-cup-2k26-predictor.onrender.com)**

---

## 🛠️ Tools & Libraries Used

- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Flask
- Joblib
- Render (Deployment)

---

## 📂 Project Structure

    FIFA_WC'26/
    │
    ├── app.py                  # Flask web application
    ├── clean_matches.csv       # Preprocessed historical match data
    ├── feature_columns.pkl     # Saved feature column names
    ├── feature_scaler.pkl      # MinMaxScaler for input features
    ├── model_xgb.pkl           # Trained XGBoost model
    ├── team_feature.pkl        # Per-team feature DataFrame
    ├── requirements.txt        # All dependencies
    └── templates/
        └── index.html          # Frontend UI

---

## 🧹 Data Preprocessing

- Collected 49,000+ historical international match records
- Removed irrelevant and duplicate entries
- Handled missing values
- Engineered ELO-based team strength ratings
- Applied Dixon-Coles model for score probability estimation
- Normalized features using MinMaxScaler

---

## 🤖 Model — XGBoost Classifier

**Why XGBoost?**
- Handles non-linear relationships in match data
- Robust against overfitting
- Fast training on large datasets
- Supports multi-class classification (Win / Draw / Loss)

**Input Features:**
- Team ELO ratings
- Head-to-head historical stats
- Recent form & performance trends
- Tournament stage weightings

---

## 📊 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

---

## 🏆 Tournament Coverage

| Feature | Details |
|---|---|
| Teams | 48 |
| Groups | 12 |
| Total Matches | 104 |
| Host Nations | USA 🇺🇸, Canada 🇨🇦, Mexico 🇲🇽 |
| Tournament Dates | Jun 11 – Jul 19, 2026 |

---

## ▶️ How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/Soumya2705/FIFA-_WC-26-Predictor.git
cd FIFA-_WC-26-Predictor
```
2. Create and activate virtual environment
```bash
python -m venv myenv
.\myenv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run the app
```bash
python app.py
```
5. Open in browser
```
http://127.0.0.1:5000
```


---

## 🚀 Deployment

This app is deployed on **Render** using **Gunicorn** as the production WSGI server.

- Platform: [Render](https://render.com)
- Start Command: `gunicorn app:app`

---

## 🔮 Future Scope

- Add player-level injury and form data
- Integrate live match results during the tournament
- Improve model with LightGBM / ensemble stacking
- Add confidence percentage for each prediction
- Mobile-friendly UI improvements

---

## 📌 License

This project is for educational and entertainment purposes only.
