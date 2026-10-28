# AI-Based HCC Risk Assessment

Simple ML project that estimates Hepatocellular Carcinoma (HCC) risk from patient risk factors and lab values.
Educational use only - not a medical device.

## Setup
pip install -r requirements.txt

## Run
python hcc_risk_assessment.py train               # train on synthetic data
python hcc_risk_assessment.py train --csv my.csv  # train on your own CSV (needs 'hcc' 0/1 column)
python hcc_risk_assessment.py demo                # high-risk vs low-risk example
python hcc_risk_assessment.py predict             # enter a patient interactively

Run `train` first; it creates hcc_model.joblib and the plots (roc_curve.png, confusion_matrix.png, feature_importance.png).
