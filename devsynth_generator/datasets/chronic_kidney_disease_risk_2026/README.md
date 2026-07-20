# Synthetic Chronic Kidney Disease Risk 2026

A synthetic chronic-kidney-disease dataset built for 2026: eGFR, creatinine, urine albumin-creatinine ratio, HbA1c, fasting glucose, blood pressure, lipid panel, wearable step counts, sleep, stress, diet quality and weekly exercise.

Classic kidney-disease datasets are often small, older, and missing modern lifestyle or wearable signals. This dataset contains 9,000 rows and 27 columns, generated with a modern CKD risk picture in mind. It includes kidney-specific clinical markers such as serum creatinine, BUN, eGFR, urine albumin-creatinine ratio, serum potassium and hemoglobin, along with metabolic, cardiovascular and lifestyle risk factors.

Risk factors move in medically plausible directions: age, BMI, systolic blood pressure, HbA1c, fasting blood glucose, creatinine, albuminuria, LDL, triglycerides, smoking, family history and stress push risk up; eGFR, HDL, physical activity, daily steps, good diet and healthy sleep pull it down. Sleep uses a U-shaped penalty, where both short and very long sleep increase risk. The binary CKD target prevalence is about 30%, making it balanced enough for beginner-friendly classification work.

## Files

- `chronic_kidney_disease_risk_2026.csv`: synthetic dataset with 9,000 rows and 27 columns.
- `gen.py`: reproducible Python generator used to create the CSV.

## Columns

- `patient_id`: Synthetic patient identifier.
- `age`: Patient age in years.
- `sex`: Female or Male.
- `bmi`: Body mass index.
- `systolic_bp`: Systolic blood pressure in mmHg.
- `diastolic_bp`: Diastolic blood pressure in mmHg.
- `fasting_blood_glucose_mg_dl`: Fasting blood glucose.
- `hba1c_percent`: HbA1c percentage.
- `serum_creatinine_mg_dl`: Serum creatinine.
- `bun_mg_dl`: Blood urea nitrogen.
- `egfr_ml_min_1_73m2`: Estimated glomerular filtration rate.
- `urine_albumin_creatinine_ratio_mg_g`: Urine albumin-creatinine ratio.
- `serum_potassium_mmol_l`: Serum potassium.
- `hemoglobin_g_dl`: Hemoglobin.
- `total_cholesterol_mg_dl`: Total cholesterol.
- `ldl_mg_dl`: LDL cholesterol.
- `hdl_mg_dl`: HDL cholesterol.
- `triglycerides_mg_dl`: Triglycerides.
- `smoking_status`: Never, Former or Current.
- `physical_activity_minutes_per_week`: Weekly physical activity minutes.
- `daily_steps`: Wearable-style daily step count.
- `sleep_hours`: Average nightly sleep.
- `stress_score`: Stress score from 1 to 10.
- `diet_quality_score`: Diet quality score from 1 to 10.
- `family_history_kidney_disease`: Family history flag.
- `ckd_risk_score`: Synthetic risk score from 0 to 100.
- `ckd_diagnosis`: Binary target, 1 for CKD risk positive and 0 otherwise.

## Ideas To Get Started

- Build a CKD classifier and compare interpretable models with gradient boosting.
- Check whether eGFR and albuminuria dominate the lifestyle columns.
- Test whether sleep has a U-shaped relationship with CKD diagnosis.
- Compare kidney-specific markers against metabolic markers such as HbA1c and BMI.
- Explore whether daily steps still matter after adjusting for age and BMI.
- Build a simple risk score and compare it to machine-learning models.

## Note On Data

This dataset is entirely synthetic. It was generated programmatically with Python for machine-learning, visualization and educational practice. It does not contain real patient records and must not be used for medical diagnosis, treatment decisions or clinical risk prediction.
