# FYP – Nutrition Waste Prediction Using AI (Random Forest, XGBoost & LSTM)

This project forecasts future nutrient waste — **Carbohydrates**, **Protein**, **Fat**, and **Fiber** — using historical weekly food waste data. It applies **Random Forest**, **XGBoost**, and **LSTM** models to predict nutrient levels for the next 8 weeks, visualized through an interactive Streamlit web interface.

---

## 🚀 Features
- Upload your weekly nutrient data and predict upcoming nutrient waste
- Visualize model forecasts across Random Forest, XGBoost, and LSTM
- Download results as CSV
- Fully modular UI with dark-themed Streamlit styling

---

## 🛠️ Project Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/FYP-Nutrition-Prediction-AI-Project.git
cd FYP-Nutrition-Prediction-AI-Project
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
```
**On Windows:**
```bash
venv\Scripts\activate
```
**On macOS/Linux:**
```bash
source venv/bin/activate
```

**If PowerShell blocks the activation:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📁 Data Structure

All processed files are located in the `/data/engineered/` directory.

### Format of lag-based training files:
| File Name            | Description                                             |
|----------------------|---------------------------------------------------------|
| `carbohydrates_lagged.csv` | Lag features (lag_1 to lag_4) + target for Carbohydrates |
| `fiber_lagged.csv`         | Lag features + target for Fiber                     |
| `protein_lagged.csv`       | Lag features + target for Protein                   |
| `fat_lagged.csv`           | Lag features + target for Fat                       |

Each row contains:
- `lag_1` to `lag_4`: Nutrient values for the past 4 weeks
- `target`: Nutrient value for the next week (week 5)

---

## 🤖 Models Used
- `Random Forest` – Scikit-learn
- `XGBoost` – XGBoost
- `LSTM` – TensorFlow/Keras (for deep learning based predictions)

---

## 📊 Output Directory
Trained models are saved in `/models/`:
- `carbohydrates_random_forest.pkl`, `fiber_lstm_model.h5`, etc.

Forecast results are saved in `/results/` after prediction:
- `carbohydrates_lstm_forecast.csv`, etc.

---

## 💻 Run the App
```bash
streamlit run app.py
```

Pages:
- **Home**: Project overview
- **Predict**: Choose model and nutrient to forecast
- **Visualize**: View past forecast results
- **Upload**: Upload your own CSV for prediction

---

## 📂 Folder Structure
```
├── app.py
├── models/                 # Saved model files
├── data/engineered/        # Preprocessed feature-lag datasets
├── results/                # Forecast outputs (CSV)
├── src/                    # Training scripts
├── pages/                  # Streamlit multipage views
│   ├── Home.py
│   ├── Predict.py
│   ├── Visualize.py
│   └── Upload.py
└── requirements.txt
```

---

## ✅ Status
All phases (preprocessing → training → forecasting → Streamlit dashboard) are complete and functional.

Need help with deployment or visual tweaks? Let me know!
