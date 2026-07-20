import argparse
import csv
import random
from pathlib import Path


COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "fasting_blood_glucose_mg_dl",
    "hba1c_percent",
    "serum_creatinine_mg_dl",
    "bun_mg_dl",
    "egfr_ml_min_1_73m2",
    "urine_albumin_creatinine_ratio_mg_g",
    "serum_potassium_mmol_l",
    "hemoglobin_g_dl",
    "total_cholesterol_mg_dl",
    "ldl_mg_dl",
    "hdl_mg_dl",
    "triglycerides_mg_dl",
    "smoking_status",
    "physical_activity_minutes_per_week",
    "daily_steps",
    "sleep_hours",
    "stress_score",
    "diet_quality_score",
    "family_history_kidney_disease",
    "ckd_risk_score",
    "ckd_diagnosis",
]


def clamp(value, low, high):
    return max(low, min(high, value))


def weighted_choice(options):
    total = sum(weight for _, weight in options)
    threshold = random.uniform(0, total)
    cumulative = 0
    for value, weight in options:
        cumulative += weight
        if threshold <= cumulative:
            return value
    return options[-1][0]


def make_row(index):
    age = int(clamp(random.gauss(52, 16), 18, 90))
    sex = weighted_choice([("Female", 0.51), ("Male", 0.49)])
    bmi = round(clamp(random.gauss(27.8, 5.8), 17.0, 48.0), 1)

    activity = int(clamp(random.gauss(145, 85) - max(0, age - 55) * 1.2, 0, 520))
    daily_steps = int(clamp(random.gauss(7200, 2600) - max(0, age - 50) * 55 + activity * 7, 900, 22000))
    sleep_hours = round(clamp(random.gauss(6.8, 1.15), 3.5, 10.5), 1)
    stress_score = int(clamp(round(random.gauss(5.2, 2.1)), 1, 10))
    diet_quality = int(clamp(round(random.gauss(6.1, 2.0) + activity / 180), 1, 10))

    smoking_status = weighted_choice([("Never", 0.58), ("Former", 0.25), ("Current", 0.17)])
    family_history = weighted_choice([("No", 0.78), ("Yes", 0.22)])

    systolic_bp = int(clamp(random.gauss(122, 16) + max(0, age - 45) * 0.35 + max(0, bmi - 27) * 0.9, 88, 205))
    diastolic_bp = int(clamp(random.gauss(78, 10) + max(0, bmi - 30) * 0.5, 52, 125))

    fasting_glucose = int(clamp(random.gauss(99, 22) + max(0, bmi - 28) * 1.4 + max(0, age - 55) * 0.22, 65, 260))
    hba1c = round(clamp(random.gauss(5.7, 0.75) + max(0, fasting_glucose - 105) / 90, 4.5, 11.8), 1)

    ldl = int(clamp(random.gauss(118, 34) + max(0, bmi - 28) * 1.8 - diet_quality * 1.3, 45, 245))
    hdl = int(clamp(random.gauss(53, 13) - max(0, bmi - 28) * 0.8 + activity / 80, 22, 100))
    triglycerides = int(clamp(random.gauss(145, 58) + max(0, bmi - 28) * 4 + max(0, fasting_glucose - 100) * 0.35, 45, 430))
    total_cholesterol = int(clamp(ldl + hdl + triglycerides / 5 + random.gauss(8, 10), 110, 360))

    base_egfr = 116 - age * 0.62
    egfr = round(clamp(base_egfr - max(0, systolic_bp - 125) * 0.18 - max(0, hba1c - 5.7) * 4.8 + random.gauss(0, 13), 12, 125), 1)
    creatinine = round(clamp((105 / max(egfr, 12)) + (0.12 if sex == "Male" else 0) + random.gauss(0, 0.12), 0.45, 6.2), 2)
    bun = int(clamp(random.gauss(15, 5) + max(0, 70 - egfr) * 0.22, 6, 70))
    uacr = int(clamp(random.lognormvariate(3.0, 0.85) + max(0, systolic_bp - 130) * 1.2 + max(0, hba1c - 6.0) * 28, 3, 1800))
    potassium = round(clamp(random.gauss(4.25, 0.38) + max(0, 45 - egfr) * 0.012, 3.1, 6.4), 1)
    hemoglobin = round(clamp(random.gauss(13.7, 1.2) - max(0, 55 - egfr) * 0.025, 8.5, 17.5), 1)

    sleep_penalty = abs(sleep_hours - 7.2) * 3.2
    risk = 0
    risk += max(0, age - 45) * 0.42
    risk += max(0, bmi - 27) * 1.25
    risk += max(0, systolic_bp - 120) * 0.38
    risk += max(0, hba1c - 5.7) * 9.0
    risk += max(0, fasting_glucose - 100) * 0.13
    risk += max(0, creatinine - 1.0) * 12.0
    risk += max(0, 90 - egfr) * 0.55
    risk += max(0, uacr - 30) * 0.035
    risk += max(0, ldl - 120) * 0.08
    risk += max(0, triglycerides - 160) * 0.035
    risk += max(0, 50 - hdl) * 0.22
    risk += stress_score * 1.25
    risk += sleep_penalty
    risk -= min(activity, 300) * 0.035
    risk -= max(0, daily_steps - 5000) * 0.0014
    risk -= diet_quality * 1.35

    if smoking_status == "Current":
        risk += 8.5
    elif smoking_status == "Former":
        risk += 3.5
    if family_history == "Yes":
        risk += 9.0

    risk += random.gauss(0, 8.5)
    risk_score = round(clamp(risk, 0, 100), 1)
    ckd_diagnosis = 1 if risk_score >= 37 else 0

    return {
        "patient_id": f"CKD2026_{index:05d}",
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "fasting_blood_glucose_mg_dl": fasting_glucose,
        "hba1c_percent": hba1c,
        "serum_creatinine_mg_dl": creatinine,
        "bun_mg_dl": bun,
        "egfr_ml_min_1_73m2": egfr,
        "urine_albumin_creatinine_ratio_mg_g": uacr,
        "serum_potassium_mmol_l": potassium,
        "hemoglobin_g_dl": hemoglobin,
        "total_cholesterol_mg_dl": total_cholesterol,
        "ldl_mg_dl": ldl,
        "hdl_mg_dl": hdl,
        "triglycerides_mg_dl": triglycerides,
        "smoking_status": smoking_status,
        "physical_activity_minutes_per_week": activity,
        "daily_steps": daily_steps,
        "sleep_hours": sleep_hours,
        "stress_score": stress_score,
        "diet_quality_score": diet_quality,
        "family_history_kidney_disease": family_history,
        "ckd_risk_score": risk_score,
        "ckd_diagnosis": ckd_diagnosis,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic 2026-style CKD risk dataset.")
    parser.add_argument("--rows", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="chronic_kidney_disease_risk_2026.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    rows = [make_row(i) for i in range(1, args.rows + 1)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(row["ckd_diagnosis"] for row in rows)
    prevalence = positives / len(rows) * 100
    print(f"Wrote {len(rows):,} rows and {len(COLUMNS)} columns to {output_path}")
    print(f"CKD diagnosis prevalence: {prevalence:.1f}%")


if __name__ == "__main__":
    main()
