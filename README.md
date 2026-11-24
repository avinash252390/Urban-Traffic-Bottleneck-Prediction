# Urban Traffic Bottleneck Prediction Website

A Streamlit website based on the uploaded project PDF. It supports CSV upload, data preview, KNN and Random Forest training, GridSearchCV tuning, model comparison, and interactive congestion severity prediction.

## 1. Requirements
- Python 3.10 or newer
- VS Code (recommended)

## 2. Open the project
Extract this ZIP, open the `traffic_bottleneck_website` folder in VS Code.

## 3. Create and activate a virtual environment (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## 4. Install packages
```powershell
python -m pip install -r requirements.txt
```

## 5. Run the website
```powershell
streamlit run app.py
```
Streamlit opens the website in your browser (usually http://localhost:8501).

## 6. Train your models
1. Open **Train & Evaluate** in the website.
2. Upload your traffic CSV file.
3. Select the congestion severity target column (usually `severity`).
4. Select input features. The project appendix assumes:
   - `vehicle_count`
   - `average_speed`
   - `hour`
   - `day_of_week`
   - `road_id`
5. Click **Train KNN and Random Forest**.
6. Review accuracy, precision, recall, F1-score, best parameters and classification reports.
7. Open **Predict Severity**, enter traffic conditions and predict.

## Important dataset note
The uploaded project PDF says the exact dataset schema is not embedded. Its example assumes the columns listed above. If your CSV uses different column names, select the matching columns in the app. The target column must contain the known severity labels (for example Low, Moderate, High, Severe).

This website uses your uploaded labeled dataset. It does not claim to provide live traffic data. Predictions are only meaningful when the model has been trained on suitable real traffic records.
