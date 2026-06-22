from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
import os
import math

def predict_custom_stack(bundle: dict, df: pd.DataFrame) -> float:
    base_preds = np.column_stack([bm.predict(df) for bm in bundle["base_models"].values()])
    return float(bundle["meta_model"].predict(base_preds)[0])

app = FastAPI(
    title="eGFR Prediction API",
    description="Stacking-model API for predicting current and projected eGFR at 180 and 360 days.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")

model_baseline = None
model_180 = None
model_360 = None

try:
    model_baseline = joblib.load(
        os.path.join(MODELS_DIR, "stacking_model_current_eGFR.pkl")
    )
    model_180 = joblib.load(
        os.path.join(MODELS_DIR, "stacking_model_eGFR_180mean.pkl")
    )
    model_360 = joblib.load(
        os.path.join(MODELS_DIR, "stacking_model_eGFR_360mean.pkl")
    )
except Exception as e:
    print("Error loading models:", e)


# Evidence-based medication effects on model input features.
# Sources: KDIGO 2022/2024 guidelines, ADA-KDIGO consensus, landmark CKD trials.
# Each entry: deltas → feature changes applied to the model; evidence → guideline/trial metadata.
MEDICATION_CATALOG: Dict[str, Dict[str, Any]] = {
    # ── Lipid-Lowering ─────────────────────────────────────────────────────────
    "atorvastatin": {
        "name": "Atorvastatin",
        "drug_class": "Statin (Moderate-Intensity)",
        "category": "lipid",
        "mechanism": "Inhibits HMG-CoA reductase, reducing hepatic cholesterol synthesis",
        "deltas": {"totalCholesterol_pct": -30, "hdl_pct": 5},
        "evidence": {
            "guideline": "KDIGO Lipid Management in CKD",
            "guideline_year": 2013,
            "recommendation": "Treat adults ≥50 yrs with CKD not on dialysis with statin or statin/ezetimibe (Grade 1B)",
            "landmark_trial": "SHARP",
            "trial_year": 2011,
            "sample_size": 9270,
            "ckd_population": "Pre-dialysis CKD and dialysis patients",
            "main_findings": "17% RRR in major atherosclerotic events (simvastatin/ezetimibe vs placebo)",
            "evidence_level": "1B",
        },
    },
    "rosuvastatin": {
        "name": "Rosuvastatin",
        "drug_class": "Statin (High-Intensity)",
        "category": "lipid",
        "mechanism": "High-potency HMG-CoA reductase inhibitor; greater LDL reduction than moderate statins",
        "deltas": {"totalCholesterol_pct": -40, "hdl_pct": 8},
        "evidence": {
            "guideline": "KDIGO Lipid Management in CKD",
            "guideline_year": 2013,
            "recommendation": "Treat adults ≥50 yrs with CKD not on dialysis with statin or statin/ezetimibe (Grade 1B)",
            "landmark_trial": "SHARP",
            "trial_year": 2011,
            "sample_size": 9270,
            "ckd_population": "Pre-dialysis CKD and dialysis patients",
            "main_findings": "17% RRR in major atherosclerotic events; high-intensity statins achieve greater LDL reduction",
            "evidence_level": "1B",
        },
    },

    # ── Antihypertensives ──────────────────────────────────────────────────────
    "lisinopril": {
        "name": "Lisinopril",
        "drug_class": "ACE Inhibitor",
        "category": "antihypertensive",
        "mechanism": "Inhibits angiotensin-converting enzyme; reduces BP, proteinuria, and glomerular pressure",
        "deltas": {"hypertension_control": True},
        "evidence": {
            "guideline": "KDIGO 2024 CKD Clinical Practice Guideline",
            "guideline_year": 2024,
            "recommendation": "First-line RAAS blockade (ACEi or ARB) for CKD with diabetes and/or albuminuria (Grade 1A)",
            "landmark_trial": "AIPRI",
            "trial_year": 1996,
            "sample_size": 583,
            "ckd_population": "Non-diabetic CKD, serum creatinine 1.5–5.0 mg/dL",
            "main_findings": "53% RRR in doubling of serum creatinine or ESKD with benazepril (ACEi) vs placebo",
            "evidence_level": "1A",
        },
    },
    "losartan": {
        "name": "Losartan",
        "drug_class": "ARB",
        "category": "antihypertensive",
        "mechanism": "Blocks angiotensin II (AT1) receptor; reduces BP, renal fibrosis, and proteinuria",
        "deltas": {"hypertension_control": True},
        "evidence": {
            "guideline": "KDIGO 2024 CKD Clinical Practice Guideline",
            "guideline_year": 2024,
            "recommendation": "First-line RAAS blockade (ACEi or ARB) for CKD with diabetes and/or albuminuria (Grade 1A)",
            "landmark_trial": "RENAAL",
            "trial_year": 2001,
            "sample_size": 1513,
            "ckd_population": "Type 2 diabetes + nephropathy, serum creatinine 1.3–3.0 mg/dL",
            "main_findings": "16% RRR in composite ESKD, doubling of serum creatinine, or death with losartan vs placebo",
            "evidence_level": "1A",
        },
    },
    "amlodipine": {
        "name": "Amlodipine",
        "drug_class": "Calcium Channel Blocker",
        "category": "antihypertensive",
        "mechanism": "Blocks L-type calcium channels; vasodilation reduces SBP by 5–10+ mmHg",
        "deltas": {"hypertension_control": True},
        "evidence": {
            "guideline": "KDIGO 2021 Blood Pressure in CKD",
            "guideline_year": 2021,
            "recommendation": "Target SBP <120 mmHg in CKD without dialysis when tolerated; CCBs are preferred add-on agents (Grade 2B)",
            "landmark_trial": "ACCOMPLISH",
            "trial_year": 2008,
            "sample_size": 11506,
            "ckd_population": "High-CV-risk patients including CKD (eGFR <60); 60% of cohort",
            "main_findings": "19.6% RRR in CV events/death: amlodipine+benazepril superior to HCTZ+benazepril",
            "evidence_level": "2B",
        },
    },
    "chlorthalidone": {
        "name": "Chlorthalidone",
        "drug_class": "Thiazide-like Diuretic",
        "category": "antihypertensive",
        "mechanism": "Inhibits NaCl cotransporter; volume reduction lowers SBP by ~10.5 mmHg in CKD",
        "deltas": {"hypertension_control": True, "bmi_pct": -1},
        "evidence": {
            "guideline": "KDIGO 2021 Blood Pressure in CKD",
            "guideline_year": 2021,
            "recommendation": "Thiazide-type diuretics recommended for BP control in CKD eGFR ≥30 (Grade 2B); chlorthalidone may be used in advanced CKD",
            "landmark_trial": "CLICK",
            "trial_year": 2021,
            "sample_size": 160,
            "ckd_population": "CKD eGFR 15–30 mL/min/1.73m² with uncontrolled hypertension",
            "main_findings": "SBP reduced by mean 10.5 mmHg vs placebo; significant reduction in urinary albumin",
            "evidence_level": "2B",
        },
    },

    # ── SGLT2 Inhibitors ───────────────────────────────────────────────────────
    "empagliflozin": {
        "name": "Empagliflozin",
        "drug_class": "SGLT2 Inhibitor",
        "category": "glycemic",
        "mechanism": "Blocks renal SGLT2; reduces intraglomerular pressure, glycosuria, and body weight",
        "deltas": {"diabetes_control": True, "bmi_pct": -2},
        "evidence": {
            "guideline": "KDIGO 2022 Diabetes Management in CKD",
            "guideline_year": 2022,
            "recommendation": "SGLT2 inhibitor recommended for T2DM + CKD with eGFR ≥20 to reduce disease progression (Grade 1A)",
            "landmark_trial": "EMPA-KIDNEY",
            "trial_year": 2022,
            "sample_size": 6609,
            "ckd_population": "CKD eGFR 20–44.9 or 45–89.9 with UACR ≥200; with or without T2DM",
            "main_findings": "22% RRR in kidney disease progression or CV death vs placebo",
            "evidence_level": "1A",
        },
    },
    "dapagliflozin": {
        "name": "Dapagliflozin",
        "drug_class": "SGLT2 Inhibitor",
        "category": "glycemic",
        "mechanism": "SGLT2 blockade reduces glomerular hyperfiltration, body weight, and HbA1c",
        "deltas": {"diabetes_control": True, "bmi_pct": -2, "hemoglobin_abs": 0.5, "hemoglobin_cap": 14.0},
        "evidence": {
            "guideline": "KDIGO 2022 Diabetes Management in CKD",
            "guideline_year": 2022,
            "recommendation": "SGLT2 inhibitor recommended for T2DM + CKD with eGFR ≥20 to reduce disease progression (Grade 1A)",
            "landmark_trial": "DAPA-CKD",
            "trial_year": 2020,
            "sample_size": 4304,
            "ckd_population": "CKD eGFR 25–75, UACR 200–5000 mg/g; ~33% without T2DM",
            "main_findings": "39% RRR in composite worsening renal function, ESKD, or renal/CV death vs placebo",
            "evidence_level": "1A",
        },
    },
    "canagliflozin": {
        "name": "Canagliflozin",
        "drug_class": "SGLT2 Inhibitor",
        "category": "glycemic",
        "mechanism": "SGLT2 inhibition lowers HbA1c, reduces body weight, and decreases intraglomerular pressure",
        "deltas": {"diabetes_control": True, "bmi_pct": -2.5},
        "evidence": {
            "guideline": "KDIGO 2022 Diabetes Management in CKD",
            "guideline_year": 2022,
            "recommendation": "SGLT2 inhibitor recommended for T2DM + CKD with eGFR ≥20 to reduce disease progression (Grade 1A)",
            "landmark_trial": "CREDENCE",
            "trial_year": 2019,
            "sample_size": 4401,
            "ckd_population": "Type 2 diabetes + CKD eGFR 30–89.9, UACR ≥300 mg/g",
            "main_findings": "30% RRR in composite ESKD, doubling serum creatinine, or renal/CV death vs placebo",
            "evidence_level": "1A",
        },
    },

    # ── MRA ────────────────────────────────────────────────────────────────────
    "finerenone": {
        "name": "Finerenone",
        "drug_class": "Mineralocorticoid Receptor Antagonist",
        "category": "antihypertensive",
        "mechanism": "Non-steroidal MRA; reduces aldosterone-driven renal inflammation and fibrosis",
        "deltas": {},
        "evidence": {
            "guideline": "KDIGO 2024 CKD Clinical Practice Guideline",
            "guideline_year": 2024,
            "recommendation": "Add finerenone to RAAS blockade in T2DM + CKD with eGFR ≥25 and elevated UACR (Grade 1A)",
            "landmark_trial": "FIDELIO-DKD",
            "trial_year": 2020,
            "sample_size": 5734,
            "ckd_population": "Type 2 diabetes + CKD, eGFR 25–75, UACR ≥30 mg/g on max-tolerated RAAS therapy",
            "main_findings": "18% RRR in composite kidney failure, sustained eGFR decline ≥40%, or renal death",
            "evidence_level": "1A",
        },
    },

    # ── Glycemic Agents ────────────────────────────────────────────────────────
    "semaglutide": {
        "name": "Semaglutide",
        "drug_class": "GLP-1 Receptor Agonist",
        "category": "glycemic",
        "mechanism": "Stimulates GLP-1 receptors; reduces appetite, body weight, and HbA1c",
        "deltas": {"diabetes_control": True, "bmi_pct": -8},
        "evidence": {
            "guideline": "KDIGO 2022 Diabetes Management in CKD",
            "guideline_year": 2022,
            "recommendation": "GLP-1 RA recommended for T2DM + CKD to reduce CV risk and support glycemic control (Grade 1B)",
            "landmark_trial": "FLOW",
            "trial_year": 2024,
            "sample_size": 3533,
            "ckd_population": "Type 2 diabetes + CKD eGFR 50–75 or <50 with albuminuria",
            "main_findings": "24% RRR in kidney disease progression, CV death, or death from kidney failure",
            "evidence_level": "1B",
        },
    },
    "metformin": {
        "name": "Metformin",
        "drug_class": "Biguanide",
        "category": "glycemic",
        "mechanism": "Activates AMPK; reduces hepatic glucose output, improves insulin sensitivity, modestly lowers LDL",
        "deltas": {"diabetes_control": True, "bmi_pct": -2, "totalCholesterol_pct": -5},
        "evidence": {
            "guideline": "ADA-KDIGO Diabetes Management in CKD Consensus",
            "guideline_year": 2022,
            "recommendation": "Continue metformin in CKD eGFR ≥30; withhold if eGFR <30 (Grade 1B). First-line agent for T2DM.",
            "landmark_trial": "UKPDS 34",
            "trial_year": 1998,
            "sample_size": 1704,
            "ckd_population": "Overweight T2DM patients; foundational glycemic control evidence",
            "main_findings": "39% RRR in diabetes-related endpoints; 36% reduction in all-cause mortality vs diet control",
            "evidence_level": "1B",
        },
    },
    "insulin": {
        "name": "Insulin",
        "drug_class": "Insulin (Various Formulations)",
        "category": "glycemic",
        "mechanism": "Exogenous insulin replacement; potent glycemic control; promotes weight gain",
        "deltas": {"diabetes_control": True, "bmi_pct": 3},
        "evidence": {
            "guideline": "ADA Standards of Medical Care in Diabetes 2026",
            "guideline_year": 2026,
            "recommendation": "Insulin required when HbA1c target cannot be achieved with non-insulin agents or in CKD eGFR <30 (Grade 1A)",
            "landmark_trial": "ADA-KDIGO Consensus 2022",
            "trial_year": 2022,
            "sample_size": 0,
            "ckd_population": "T2DM + CKD across all eGFR stages; insulin dose adjustment required in advanced CKD",
            "main_findings": "Effective HbA1c reduction; weight gain ~2–4 kg; hypoglycemia risk higher in CKD due to reduced renal insulin clearance",
            "evidence_level": "1A",
        },
    },

    # ── CKD Anemia Management ──────────────────────────────────────────────────
    "darbepoetin": {
        "name": "Darbepoetin Alfa",
        "drug_class": "Erythropoiesis-Stimulating Agent (ESA)",
        "category": "anemia",
        "mechanism": "Long-acting EPO analogue; stimulates RBC production to correct CKD anemia",
        "deltas": {"hemoglobin_abs": 2.5, "hemoglobin_cap": 12.0},
        "evidence": {
            "guideline": "KDIGO Anemia in CKD Guideline",
            "guideline_year": 2012,
            "recommendation": "Use ESA to avoid transfusion; target Hb 10–11.5 g/dL. Do not target near-normal Hb (Grade 1A).",
            "landmark_trial": "TREAT",
            "trial_year": 2009,
            "sample_size": 4038,
            "ckd_population": "Type 2 diabetes + CKD not on dialysis, Hb <11 g/dL",
            "main_findings": "Targeting Hb ~13 g/dL doubled stroke risk vs reactive strategy; Hb target 10–11.5 g/dL recommended",
            "evidence_level": "1A",
        },
    },
    "epoetin": {
        "name": "Epoetin Alfa",
        "drug_class": "Erythropoiesis-Stimulating Agent (ESA)",
        "category": "anemia",
        "mechanism": "Recombinant human EPO; stimulates erythropoiesis to treat CKD-related anemia",
        "deltas": {"hemoglobin_abs": 2.0, "hemoglobin_cap": 12.0},
        "evidence": {
            "guideline": "KDIGO Anemia in CKD Guideline",
            "guideline_year": 2012,
            "recommendation": "Use ESA to avoid transfusion; target Hb 10–11.5 g/dL. Do not target normal Hb (Grade 1A).",
            "landmark_trial": "CHOIR",
            "trial_year": 2006,
            "sample_size": 1432,
            "ckd_population": "CKD not on dialysis with anemia",
            "main_findings": "Targeting Hb 13.5 g/dL increased composite CV endpoint vs 11.3 g/dL target; reinforced conservative Hb targets",
            "evidence_level": "1A",
        },
    },
    "iv_iron": {
        "name": "IV Iron Sucrose",
        "drug_class": "Iron Supplement",
        "category": "anemia",
        "mechanism": "Replenishes iron stores for hemoglobin synthesis; reduces ESA requirements",
        "deltas": {"hemoglobin_abs": 1.5, "hemoglobin_cap": 12.0},
        "evidence": {
            "guideline": "KDIGO Anemia in CKD Guideline",
            "guideline_year": 2012,
            "recommendation": "IV iron supplementation for iron-deficient CKD anemia when Hb response is needed and oral iron is insufficient (Grade 1C)",
            "landmark_trial": "PIVOTAL",
            "trial_year": 2019,
            "sample_size": 2141,
            "ckd_population": "Hemodialysis patients with CKD anemia",
            "main_findings": "Proactive high-dose IV iron reduced ESA dose by 19% and reduced composite CV events vs reactive low-dose strategy",
            "evidence_level": "1C",
        },
    },

    # ── Combinations ───────────────────────────────────────────────────────────
    "acei_statin": {
        "name": "ACEi + Statin Bundle",
        "drug_class": "Combination (Lisinopril + Atorvastatin)",
        "category": "combination",
        "mechanism": "Standard CKD cardioprotective bundle: RAAS blockade for BP/proteinuria + lipid lowering",
        "deltas": {"hypertension_control": True, "totalCholesterol_pct": -30, "hdl_pct": 5},
        "evidence": {
            "guideline": "KDIGO 2024 CKD + Lipid Management Guidelines",
            "guideline_year": 2024,
            "recommendation": "RAAS blockade + statin therapy is the evidence-based standard of care for CKD with hypertension and dyslipidemia",
            "landmark_trial": "AIPRI + SHARP",
            "trial_year": 2011,
            "sample_size": 9853,
            "ckd_population": "Non-diabetic and diabetic CKD with hypertension and dyslipidemia",
            "main_findings": "Combined: 53% RRR in renal progression (AIPRI) + 17% RRR in atherosclerotic events (SHARP)",
            "evidence_level": "1A",
        },
    },
    "sglt2_statin": {
        "name": "SGLT2i + Statin Bundle",
        "drug_class": "Combination (Empagliflozin + Atorvastatin)",
        "category": "combination",
        "mechanism": "Dual renal and cardiometabolic protection: SGLT2 blockade + lipid lowering",
        "deltas": {"diabetes_control": True, "bmi_pct": -2, "totalCholesterol_pct": -30, "hdl_pct": 5},
        "evidence": {
            "guideline": "KDIGO 2022 + Lipid Management Guidelines",
            "guideline_year": 2022,
            "recommendation": "Emerging standard of care: SGLT2i for renoprotection + statin for CV risk reduction in CKD + T2DM",
            "landmark_trial": "EMPA-KIDNEY + SHARP",
            "trial_year": 2022,
            "sample_size": 15879,
            "ckd_population": "CKD with or without T2DM and dyslipidemia",
            "main_findings": "Combined: 22% RRR in renal/CV death (EMPA-KIDNEY) + 17% RRR in atherosclerotic events (SHARP)",
            "evidence_level": "1A",
        },
    },
}


class PatientData(BaseModel):
    gender: str           # "M" or "F"
    age: float
    bmi: float
    hemoglobin: float
    hypertension: bool
    diabetes: bool
    totalCholesterol: float
    hdl: float
    serumCreatinine: Optional[float] = None   # if provided, CKD-EPI is used for baseline eGFR


class MedicationRequest(PatientData):
    medication_id: str


def calculate_ckd_epi_2021(scr: float, age: float, is_female: bool) -> float:
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    min_ratio = min(scr / kappa, 1.0)
    max_ratio = max(scr / kappa, 1.0)
    egfr = 142 * (min_ratio ** alpha) * (max_ratio ** -1.200) * (0.9938 ** age)
    if is_female:
        egfr *= 1.012
    return egfr


def run_prediction(data: PatientData) -> dict:
    """Run the full eGFR prediction pipeline and return {egfr, model_used, projected_6m, projected_12m}."""
    gender = 1 if "M" in data.gender.upper() else 0
    dm    = 1 if data.diabetes else 0
    htn   = 1 if data.hypertension else 0
    age   = data.age
    bmi   = data.bmi
    hb    = data.hemoglobin
    tc    = data.totalCholesterol
    hdl   = data.hdl

    # ── Baseline eGFR ──────────────────────────────────────────────────────
    if data.serumCreatinine is not None and data.serumCreatinine > 0:
        egfr_val = calculate_ckd_epi_2021(data.serumCreatinine, age, gender == 0)
        model_used = "ckd_epi_2021"
    elif model_baseline is not None:
        row = {
            "Gender":                gender,
            "Hypertension_Status":   htn,
            "DM_episode":            dm,
            "Chol_HDL_ratio":        tc / hdl if hdl > 0 else 0.0,
            "log_BMI":               math.log1p(max(bmi, 0.0)),
            "Age":                   age,
            "Age_sq":                age ** 2,
            "Age_Hb":                age * hb,
            "DMxHTN":                dm * htn,
            "log_Hemoglobin":        math.log1p(max(hb,  0.0)),
            "log_Total Cholesterol": math.log1p(max(tc,  0.0)),
            "log_HDL Cholesterol":   math.log1p(max(hdl, 0.0)),
        }
        BASELINE_COLS = [
            "Gender", "Hypertension_Status", "DM_episode", "Chol_HDL_ratio",
            "log_BMI", "Age", "Age_sq", "Age_Hb", "DMxHTN",
            "log_Hemoglobin", "log_Total Cholesterol", "log_HDL Cholesterol",
        ]
        df = pd.DataFrame([row])[BASELINE_COLS]
        egfr_val = predict_custom_stack(model_baseline, df)
        model_used = "stacking_baseline"
    else:
        raise HTTPException(status_code=500, detail="Baseline model not loaded.")

    # ── Projection features ────────────────────────────────────────────────
    PROJ_COLS = [
        "Gender", "AgeBaseline", "Body Mass Index", "Hemoglobin",
        "Hypertension_Status", "DM_episode",
        "eGFR_Age_ratio", "eGFR_Hb_product", "eGFR_BMI_ratio", "eGFR_Hb_Age",
    ]
    proj_row = {
        "Gender":              gender,
        "AgeBaseline":         age,
        "Body Mass Index":     bmi,
        "Hemoglobin":          hb,
        "Hypertension_Status": htn,
        "DM_episode":          dm,
        "eGFR_Age_ratio":  math.log1p(max(egfr_val / age,      0.0) if age > 0 else 0.0),
        "eGFR_Hb_product": math.log1p(max(egfr_val * hb,       0.0)),
        "eGFR_BMI_ratio":  math.log1p(max(egfr_val / bmi,      0.0) if bmi > 0 else 0.0),
        "eGFR_Hb_Age":     math.log1p(max(egfr_val * hb / age, 0.0) if age > 0 else 0.0),
    }

    projected_180 = None
    projected_360 = None

    if model_180 is not None:
        projected_180 = float(model_180["model"].predict(pd.DataFrame([proj_row])[PROJ_COLS])[0])
    if model_360 is not None:
        projected_360 = float(model_360["model"].predict(pd.DataFrame([proj_row])[PROJ_COLS])[0])

    return {
        "egfr":          egfr_val,
        "model_used":    model_used,
        "projected_6m":  projected_180,
        "projected_12m": projected_360,
    }


def apply_medication_deltas(data: PatientData, deltas: dict) -> PatientData:
    """Return a new PatientData with medication-induced feature changes applied."""
    bmi  = data.bmi
    hb   = data.hemoglobin
    tc   = data.totalCholesterol
    hdl  = data.hdl
    htn  = data.hypertension
    dm   = data.diabetes

    if "bmi_pct" in deltas:
        bmi = max(bmi * (1 + deltas["bmi_pct"] / 100), 10.0)

    if "hemoglobin_abs" in deltas:
        hb = hb + deltas["hemoglobin_abs"]
        cap = deltas.get("hemoglobin_cap", 18.0)
        hb = min(hb, cap)

    if "totalCholesterol_pct" in deltas:
        tc = max(tc * (1 + deltas["totalCholesterol_pct"] / 100), 50.0)

    if "hdl_pct" in deltas:
        hdl = hdl * (1 + deltas["hdl_pct"] / 100)

    if deltas.get("hypertension_control") and htn:
        htn = False

    if deltas.get("diabetes_control") and dm:
        dm = False

    return PatientData(
        gender=data.gender,
        age=data.age,
        bmi=bmi,
        hemoglobin=hb,
        hypertension=htn,
        diabetes=dm,
        totalCholesterol=tc,
        hdl=hdl,
        serumCreatinine=data.serumCreatinine,
    )


@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/health")


@app.get("/health", tags=["System Diagnostics"], summary="Check API Health & Model Status")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "baseline_egfr": model_baseline is not None,
            "projection_180d": model_180 is not None,
            "projection_360d": model_360 is not None,
        },
    }


@app.get("/medications", tags=["Medication Simulator"], summary="List available medications")
def list_medications():
    return [
        {
            "id": med_id,
            "name": med["name"],
            "drug_class": med["drug_class"],
            "category": med["category"],
            "mechanism": med["mechanism"],
        }
        for med_id, med in MEDICATION_CATALOG.items()
    ]


@app.post("/predict", tags=["eGFR Prediction"], summary="Predict Current and Future eGFR")
def predict_egfr(data: PatientData):
    try:
        return run_prediction(data)
    except HTTPException:
        raise
    except Exception as e:
        print("Prediction Error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/simulate_medication", tags=["Medication Simulator"], summary="Simulate eGFR change under a medication")
def simulate_medication(data: MedicationRequest):
    try:
        med = MEDICATION_CATALOG.get(data.medication_id)
        if med is None:
            raise HTTPException(status_code=404, detail=f"Unknown medication_id: {data.medication_id}")

        base_data = PatientData(**{k: v for k, v in data.model_dump().items() if k != "medication_id"})

        baseline_result = run_prediction(base_data)
        sim_data = apply_medication_deltas(base_data, med["deltas"])
        simulated_result = run_prediction(sim_data)

        # Build feature delta summary for the UI
        feature_deltas = {}
        d = med["deltas"]
        if "bmi_pct" in d:
            feature_deltas["bmi"] = {"before": base_data.bmi, "after": round(sim_data.bmi, 2), "pct": d["bmi_pct"]}
        if "hemoglobin_abs" in d:
            feature_deltas["hemoglobin"] = {"before": base_data.hemoglobin, "after": round(sim_data.hemoglobin, 2), "abs": d["hemoglobin_abs"]}
        if "totalCholesterol_pct" in d:
            feature_deltas["totalCholesterol"] = {"before": base_data.totalCholesterol, "after": round(sim_data.totalCholesterol, 2), "pct": d["totalCholesterol_pct"]}
        if "hdl_pct" in d:
            feature_deltas["hdl"] = {"before": base_data.hdl, "after": round(sim_data.hdl, 2), "pct": d["hdl_pct"]}
        if d.get("hypertension_control") and base_data.hypertension != sim_data.hypertension:
            feature_deltas["hypertension"] = {"before": base_data.hypertension, "after": sim_data.hypertension, "note": "controlled"}
        if d.get("diabetes_control") and base_data.diabetes != sim_data.diabetes:
            feature_deltas["diabetes"] = {"before": base_data.diabetes, "after": sim_data.diabetes, "note": "controlled"}

        egfr_delta = simulated_result["egfr"] - baseline_result["egfr"]
        will_improve = egfr_delta > 2.0

        return {
            "medication_id":   data.medication_id,
            "medication_name": med["name"],
            "drug_class":      med["drug_class"],
            "mechanism":       med["mechanism"],
            "baseline":        baseline_result,
            "simulated":       simulated_result,
            "egfr_delta":      round(egfr_delta, 2),
            "will_improve":    will_improve,
            "feature_deltas":  feature_deltas,
            "evidence":        med.get("evidence"),
        }

    except HTTPException:
        raise
    except Exception as e:
        print("Simulation Error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8001)
