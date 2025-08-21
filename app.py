!pip install pandas numpy scikit-learn gradio fpdf

import pandas as pd
import numpy as np
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fpdf import FPDF
import tempfile
import traceback
from datetime import datetime

# ==================== MEDICAL CONSTANTS ====================
NORMAL_RANGES = {
    'Glucose': (70, 140), 'BMI': (18.5, 24.9),
    'Cholesterol': (125, 200), 'BP_Systolic': (90, 120),
    'Bilirubin': (0.1, 1.2), 'ALT': (7, 56),
    'Creatinine': (0.6, 1.2), 'TSH': (0.4, 4.0),
    'LDL': (0, 130), 'HDL': (40, 60),
    'Uric Acid': (3.4, 7.0), 'Triglycerides': (0, 150)
}

# ==================== DISEASE MODELS ====================
class DiseaseModel:
    def __init__(self, n_features):  # FIXED init -> __init__
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Dummy training with random data
        X = np.random.rand(100, n_features)
        y = np.random.randint(0, 2, 100)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, inputs):
        if len(inputs) != self.scaler.n_features_in_:
            raise ValueError(f"Expected {self.scaler.n_features_in_} features, got {len(inputs)}")
        return self.model.predict(self.scaler.transform([inputs]))[0]

models = {
    'diabetes': DiseaseModel(8),
    'heart': DiseaseModel(13),
    'liver': DiseaseModel(10),
    'kidney': DiseaseModel(10),
    'parkinsons': DiseaseModel(22),
    'lipid': DiseaseModel(5),
    'uric': DiseaseModel(3),
    'thyroid': DiseaseModel(5)
}

# ==================== PROFESSIONAL PDF REPORT ====================
class MedicalReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'MedScan Pro Report', 0, 1, 'C')
        self.ln(10)
        
    def add_parameter(self, name, value, unit=""):
        self.set_font('Arial', '', 12)
        if name in NORMAL_RANGES:
            low, high = NORMAL_RANGES[name]
            status = "Normal" if low <= value <= high else "Abnormal"
            self.set_text_color(255,0,0) if status == "Abnormal" else self.set_text_color(0,155,0)
            self.cell(0, 10, f'{name}: {value}{unit} ({low}-{high})', 0, 1)
        else:
            self.cell(0, 10, f'{name}: {value}{unit}', 0, 1)
        self.set_text_color(0,0,0)

def generate_report(patient_info, inputs, predictions):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf = MedicalReport()
            pdf.add_page()
            
            # Patient Information
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f'Patient: {patient_info["name"]}', 0, 1)
            pdf.add_parameter('Age', patient_info['age'], ' years')
            pdf.cell(0, 10, f'Sex: {"Male" if patient_info["sex"] == 1 else "Female"}', 0, 1)
            
            # Lab Results
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Key Laboratory Findings:', 0, 1)
            for param, value in inputs.items():
                pdf.add_parameter(param, value)
            
            # Clinical Predictions
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Diagnostic Summary:', 0, 1)
            for disease, status in predictions.items():
                color = (255,0,0) if status in ['Positive', 'High Risk', 'Detected'] else (0,155,0)
                pdf.set_text_color(*color)
                pdf.cell(0, 10, f'{disease}: {status}', 0, 1)
            
            pdf.output(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        print(f"PDF generation error: {str(e)}")
        return None

# ==================== PREDICTION ENGINE ====================
def predict_health(
    name, age, sex,
    # Diabetes
    pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf,
    # Heart
    cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,
    # Liver & Kidney
    bilirubin, alkphos, sgpt, sgot, tp, alb, ag_ratio, liver_creatinine,
    kidney_bp, kidney_creatinine, gfr, bun, sodium, potassium, kidney_alb, kidney_hdl, kidney_triglycerides,
    # Parkinson's
    parkinsons_features,
    # Lipid & Uric
    total_chol, ldl, hdl, triglycerides,
    uric_acid
):
    try:
        # Collect lab inputs
        inputs = {
            'Glucose': glucose,
            'Cholesterol': chol,
            'BP_Systolic': bp,
            'Bilirubin': bilirubin,
            'Creatinine': kidney_creatinine,
            'TSH': 2.5,  # Example value
            'LDL': ldl,
            'HDL': hdl,
            'Uric Acid': uric_acid,
            'Triglycerides': triglycerides
        }
        
        predictions = {}
        
        # Diabetes Prediction
        diabetes_input = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
        predictions['Diabetes'] = 'Positive' if models['diabetes'].predict(diabetes_input) else 'Negative'
        
        # Heart Disease Prediction
        heart_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        predictions['Heart Disease'] = 'High Risk' if models['heart'].predict(heart_input) else 'Low Risk'
        
        # Liver Disease Prediction
        liver_input = [age, sex, bilirubin, alkphos, sgpt, sgot, tp, alb, ag_ratio, liver_creatinine]
        predictions['Liver Disease'] = 'Detected' if models['liver'].predict(liver_input) else 'Normal'
        
        # Kidney Disease Prediction
        kidney_input = [age, kidney_bp, kidney_creatinine, gfr, bun, sodium, potassium, kidney_alb, kidney_hdl, kidney_triglycerides]
        predictions['Kidney Disease'] = 'Detected' if models['kidney'].predict(kidney_input) else 'Normal'
        
        # Parkinson's Disease Prediction
        try:
            parkinsons_values = list(map(float, parkinsons_features.split(",")))
            if len(parkinsons_values) == 22:
                predictions["Parkinson's"] = 'Detected' if models['parkinsons'].predict(parkinsons_values) else 'Normal'
        except:
            predictions["Parkinson's"] = "Invalid input"
        
        # Lipid Profile
        lipid_input = [total_chol, ldl, hdl, triglycerides, age]
        predictions['Lipid Disorder'] = 'Abnormal' if models['lipid'].predict(lipid_input) else 'Normal'
        
        # Uric Acid
        uric_input = [uric_acid, age, sex]
        predictions['Hyperuricemia'] = 'Detected' if models['uric'].predict(uric_input) else 'Normal'
        
        # Generate PDF
        report_path = generate_report(
            {'name': name, 'age': age, 'sex': sex},
            inputs,
            predictions
        )
        
        return predictions, report_path
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}, None

# ==================== GRADIO INTERFACE ====================
with gr.Blocks(theme=gr.themes.Soft(), title="MedScan Pro") as app:
    gr.Markdown("# 🏥 MedScan Pro: 8-Disease Diagnostic Platform")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Patient Information")
            name = gr.Textbox(label="Full Name")
            age = gr.Number(value=45, label="Age", minimum=18, maximum=120)  # FIXED value=
            sex = gr.Radio([0, 1], label="Sex (0=Female, 1=Male)")
            
            with gr.Tab("Diabetes"):
                pregnancies = gr.Number(value=2, label="Pregnancies")
                glucose = gr.Number(value=120, label="Glucose (mg/dL)")
                bp = gr.Number(value=80, label="Blood Pressure (mmHg)")
                skin_thickness = gr.Number(value=25, label="Skin Thickness (mm)")
                insulin = gr.Number(value=80, label="Insulin (μU/mL)")
                bmi = gr.Number(value=28.5, label="BMI")
                dpf = gr.Number(value=0.5, label="Diabetes Pedigree Function")
            
            with gr.Tab("Heart Disease"):
                cp = gr.Number(value=2, label="Chest Pain Type (0-3)")
                trestbps = gr.Number(value=130, label="Resting BP (mmHg)")
                chol = gr.Number(value=200, label="Cholesterol (mg/dL)")
                fbs = gr.Number(value=0, label="Fasting Blood Sugar (>120)")
                restecg = gr.Number(value=1, label="Resting ECG (0-2)")
                thalach = gr.Number(value=150, label="Max Heart Rate")
                exang = gr.Number(value=0, label="Exercise Angina (0/1)")
                oldpeak = gr.Number(value=1.5, label="ST Depression")
                slope = gr.Number(value=2, label="Slope (0-2)")
                ca = gr.Number(value=0, label="Major Vessels (0-3)")
                thal = gr.Number(value=3, label="Thalassemia (3/6/7)")
            
            with gr.Tab("Liver & Kidney"):
                bilirubin = gr.Number(value=0.8, label="Bilirubin (mg/dL)")
                alkphos = gr.Number(value=120, label="Alkaline Phosphatase (IU/L)")
                sgpt = gr.Number(value=40, label="SGPT (ALT) (IU/L)")
                sgot = gr.Number(value=35, label="SGOT (AST) (IU/L)")
                tp = gr.Number(value=6.5, label="Total Protein (g/dL)")
                alb = gr.Number(value=3.5, label="Albumin (g/dL)")
                ag_ratio = gr.Number(value=1.0, label="A/G Ratio")
                liver_creatinine = gr.Number(value=0.8, label="Liver Creatinine (mg/dL)")
                kidney_bp = gr.Number(value=80, label="Kidney BP (mmHg)")
                kidney_creatinine = gr.Number(value=0.9, label="Kidney Creatinine (mg/dL)")
                gfr = gr.Number(value=90, label="GFR (mL/min/1.73m²)")
                bun = gr.Number(value=40, label="Blood Urea Nitrogen (mg/dL)")
                sodium = gr.Number(value=140, label="Sodium (mEq/L)")
                potassium = gr.Number(value=4.0, label="Potassium (mEq/L)")
                kidney_alb = gr.Number(value=3.5, label="Kidney Albumin (g/dL)")
                kidney_hdl = gr.Number(value=45, label="Kidney HDL (mg/dL)")
                kidney_triglycerides = gr.Number(value=150, label="Kidney Triglycerides (mg/dL)")
            
            with gr.Tab("Other Tests"):
                parkinsons_features = gr.Textbox(
                    value="119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654",
                    label="Parkinson's Features (22 comma-separated values)"
                )
                total_chol = gr.Number(value=200, label="Total Cholesterol (mg/dL)")
                ldl = gr.Number(value=130, label="LDL (mg/dL)")
                hdl = gr.Number(value=45, label="HDL (mg/dL)")
                triglycerides = gr.Number(value=150, label="Triglycerides (mg/dL)")
                uric_acid = gr.Number(value=6.8, label="Uric Acid (mg/dL)")
    
    submit_btn = gr.Button("Generate Comprehensive Report", variant="primary")
    
    with gr.Row():
        predictions_output = gr.JSON(label="Diagnostic Results")
        report_output = gr.File(label="Download Full Report")
    
    submit_btn.click(
        predict_health,
        inputs=[
            name, age, sex,
            pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf,
            cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,
            bilirubin, alkphos, sgpt, sgot, tp, alb, ag_ratio, liver_creatinine,
            kidney_bp, kidney_creatinine, gfr, bun, sodium, potassium, kidney_alb, kidney_hdl, kidney_triglycerides,
            parkinsons_features,
            total_chol, ldl, hdl, triglycerides,
            uric_acid
        ],
        outputs=[predictions_output, report_output]
    )

if __name__ == "__main__":  # FIXED
    app.launch(share=True)
