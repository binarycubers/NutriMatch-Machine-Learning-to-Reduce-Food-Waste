# FYP – Nutrition Waste Prediction Using AI (Random Forest & XGBoost)
This project predicts the future nutrient waste (Carbohydrates, Protein, Fat, Fiber) for a local store using machine learning models — Random Forest and XGBoost. The models are trained on historical weekly data derived from food waste records and forecast the next 8 weeks of nutrient waste.
---
## :package: Project Setup
### 1. Clone the repository
git clone https://github.com/your-username/FYP-Nutrition-Prediction-AI-Project.git
cd FYP-Nutrition-Prediction-AI-Project


### 2. Create and activate virtual environment
python -m venv venv

### On Windows:
venv\Scripts\activate

### On macOS/Linux:
source venv/bin/activate


### If you get a PowerShell execution policy error on Windows:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


### 3. Install dependencies
pip install -r requirements.txt

### Data Overview
All raw, processed, and engineered files are organized in the /data/ folder:
data/engineered/ contains lag-based training files:
File NameDescriptioncarb_lagged.csvLag features (lag_1 to lag_4) for Carbohydrates + target (week 5)fiber_lagged.csvLag features for Fiberprotein_lagged.csvLag features for Proteinfat_lagged.csvLag features for Fat

### Each row in these files represents:
Inputs: The last 4 weeks of nutrient values (lag_1, lag_2, lag_3, lag_4)
Target: The nutrient value for the following (5th) week
These files are used to train time series forecasting models using:
Random Forest
XGBoost
