import os 
from os import chdir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, classification_report
from seaborn import heatmap

training_file = r"C:\Users\ErikV\Desktop\VS Code\Coursework\BENG0095_T1\Dataset\diabetic_data_training.csv"
df_tr = pd.read_csv(training_file, header=0, na_values=['?'])
test_file = r"C:\Users\ErikV\Desktop\VS Code\Coursework\BENG0095_T1\Dataset\diabetic_data_test.csv"
df_test = pd.read_csv(test_file, header=0, na_values=['?'])

mv_admission_type_id = [5, 6, 8]
mv_discharge_disposition_id = [11, 18, 25, 26]
mv_admission_source_id = [9, 15, 17, 20, 21]

'''
# Replace "null-like" values with NaN
df_tr['admission_type_id'] = df_tr['admission_type_id'].replace(mv_admission_type_id, pd.NA)
df_tr['discharge_disposition_id'] = df_tr['discharge_disposition_id'].replace(mv_discharge_disposition_id, pd.NA)
df_tr['admission_source_id'] = df_tr['admission_source_id'].replace(mv_admission_source_id, pd.NA)

df_tr['admission_type_id'] = df_tr['admission_type_id'].fillna(6)
df_tr['discharge_disposition_id'] = df_tr['discharge_disposition_id'].fillna(18)
df_tr['admission_source_id'] = df_tr['admission_source_id'].fillna(17)

# Apply the same replacement for the test dataset
df_test['admission_type_id'] = df_test['admission_type_id'].fillna(6)
df_test['discharge_disposition_id'] = df_test['discharge_disposition_id'].fillna(18)
df_test['admission_source_id'] = df_test['admission_source_id'].fillna(17)
'''


'''
COLUMN DROP MODULE
'''
# Display missing value percentages
num = len(df_tr)
mis_percentage = (df_tr.isnull().sum()[df_tr.isnull().sum() > 0] / num) * 100
#print("Missing Value Percentages:\n", mis_percentage)

# Drop Unrelated && Highly Missing Columns
df_tr = df_tr.drop(['encounter_id', 'patient_nbr', 'payer_code'], axis='columns')
df_tr = df_tr.drop(['weight', 'medical_specialty', 'max_glu_serum', 'A1Cresult'], axis='columns')
df_test = df_test.drop(['encounter_id', 'patient_nbr', 'payer_code'], axis='columns')
df_test = df_test.drop(['weight', 'medical_specialty', 'max_glu_serum', 'A1Cresult'], axis='columns')

mis_percentage = (df_tr.isnull().sum()[df_tr.isnull().sum() > 0] / num) * 100
#print("Updated Missing Value Percentages:\n", mis_percentage)


'''
MEDICINE COLUMNS PROCESSING MODULE
'''
# Identify Medicine Columns
medicine_columns = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]

# Encode
for col in medicine_columns:
    if col in df_tr.columns:
        df_tr[col] = df_tr[col].replace(
            {"No": 0, "Steady": -2, "Up": 1, "Down": 3}
        )
        df_test[col] = df_test[col].replace(
            {"No": 0, "Steady": -2, "Up": 1, "Down": 3}
        )

df_tr['change'] = df_tr['change'].replace({"Ch": 1, "No": 0})
df_tr['diabetesMed'] = df_tr['diabetesMed'].replace({"Yes": 1, "No": 0})
df_test['change'] = df_test['change'].replace({"Ch": 1, "No": 0})
df_test['diabetesMed'] = df_test['diabetesMed'].replace({"Yes": 1, "No": 0})


'''
AGE MAPPING MODULE
'''
age_mapping = {
    '[0-10)': 'Young',
    '[10-20)': 'Young',
    '[20-30)': 'Adult',
    '[30-40)': 'Middle Age',
    '[40-50)': 'Middle Age',
    '[50-60)': 'Elder',
    '[60-70)': 'Elder',
    '[70-80)': 'Elder',
    '[80-90)': 'Elder',
    '[90-100)': 'Elder'
}

# Apply Mapping
df_tr['age_group'] = df_tr['age'].map(age_mapping)
df_test['age_group'] = df_test['age'].map(age_mapping)
df_tr = df_tr.drop(columns=['age'])
df_test = df_test.drop(columns=['age'])
df_tr = pd.get_dummies(df_tr, columns=['age_group'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['age_group'], drop_first=True)


'''
DIAG COLUMNS (ICD-9) PROCESSING MODULE
'''
# Categorize ICD-9 Codes
def categorize_icd9(code):
    try:
        code = float(code)
        if 1 <= code < 140:
            return "Infectious and parasitic diseases"
        elif 140 <= code < 240:
            return "Neoplasms"
        elif 240 <= code < 280:
            return "Endocrine, nutritional, and metabolic diseases"
        elif 280 <= code < 290:
            return "Blood and blood-forming organs"
        elif 290 <= code < 320:
            return "Mental disorders"
        elif 320 <= code < 390:
            return "Nervous system and sense organs"
        elif 390 <= code < 460:
            return "Circulatory system"
        elif 460 <= code < 520:
            return "Respiratory system"
        elif 520 <= code < 580:
            return "Digestive system"
        elif 580 <= code < 630:
            return "Genitourinary system"
        elif 630 <= code < 680:
            return "Complications of pregnancy, childbirth, and the puerperium"
        elif 680 <= code < 710:
            return "Skin and subcutaneous tissue"
        elif 710 <= code < 740:
            return "Musculoskeletal system and connective tissue"
        elif 740 <= code < 760:
            return "Congenital anomalies"
        elif 760 <= code < 780:
            return "Perinatal period"
        elif 780 <= code < 800:
            return "Symptoms, signs, and ill-defined conditions"
        elif 800 <= code < 1000:
            return "Injury and poisoning"
        else:
            return "Other"
    except:
        return "Unknown"

for col in ['diag_1', 'diag_2', 'diag_3']:
    df_tr[col] = df_tr[col].apply(categorize_icd9)
    df_test[col] = df_test[col].apply(categorize_icd9)

def fill_diag_missing(row):
    diags = [row['diag_1'], row['diag_2'], row['diag_3']]
    valid_diags = []

    # Store non-missing values for later index and Fill
    for d in diags:
        if d != "Unknown":
            valid_diags.append(d)

    # Fill based on the number, index of valid values
    if len(valid_diags) == 1:
        # Given 1 combination: E1 NaN NaN, or E1 at any other element --> After Refine = E1 E1 E1
        filled_diags = [valid_diags[0], valid_diags[0], valid_diags[0]]
    elif len(valid_diags) == 2:
        idx = diags.index("Unknown")
        # Index the missing element, given total 2 combinations: 1: E1 E2 NaN --> After Refine = E1 E2 E2
                                                               # 2: E1 NaN E2 --> After Refine = E1 E1 E2
                                                               # or NaN E1 E2 --> After Refine = E1 E1 E2 (E1's idx=0, same case as above)
        if idx == 2:
            filled_diags = [valid_diags[0], valid_diags[1], valid_diags[1]]
        else:
            filled_diags = [valid_diags[0], valid_diags[0], valid_diags[1]]
    else:
        filled_diags = diags

    return filled_diags

# Fill
df_tr[['diag_1', 'diag_2', 'diag_3']] = df_tr.apply(fill_diag_missing, axis=1, result_type="expand")
df_test[['diag_1', 'diag_2', 'diag_3']] = df_test.apply(fill_diag_missing, axis=1, result_type="expand")

# Encode
df_tr = pd.get_dummies(df_tr, columns=['diag_1', 'diag_2', 'diag_3'], prefix=['diag1', 'diag2', 'diag3'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['diag_1', 'diag_2', 'diag_3'], prefix=['diag1', 'diag2', 'diag3'], drop_first=True)


'''
PREPARATION MODULE
'''
X_train, y_train = df_tr.drop("readmitted", axis=1), df_tr["readmitted"].copy()
X_te, y_te = df_test.drop("readmitted", axis=1), df_test["readmitted"].copy()

categorical_features = ['race', 'gender']

X_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_encoded = X_encoded.drop("gender_Unknown/Invalid", axis=1)
X_encoded = X_encoded.drop("diag1_Other", axis=1)
y_encoded = y_train.replace({'>30': 1, 'NO': 0, '<30': 2})

X_test = pd.get_dummies(X_te, columns=categorical_features, drop_first=True)
y_test = y_te.replace({'>30': 1, 'NO': 0, '<30': 2})


'''
*备注*  X_test进行完encoding之后没有gender_Unknown/Invalid和diag1_Other这两列，故写了这行代码
        我认为这个有点奇怪，因为处理ICD-9的时候我进行了“Others”的分类，且test data里面有属于Others分类的数据
'''
X_test = X_test.reindex(columns=X_encoded.columns, fill_value=0)

# Train the Naive-Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_encoded, y_encoded)

# Predict the Test Set
y_pred = nb_model.predict(X_test)


'''
Case Check Module
'''
print("Encoded Features (X):\n", X_encoded.head())
print("Encoded Features (Test):\n", X_test.head())
print("Encoded Target (y):\n", y_encoded.head())

#print("Missing values in X_encoded:\n", X_encoded.isnull().sum())
#print("Missing values in y_encoded:\n", y_encoded.isnull().sum())

non_numeric_columns = X_encoded.select_dtypes(include=['object']).columns
#print("Non-Numeric Columns:\n", non_numeric_columns)
#print("Missing values in X_test:\n", X_test.isnull().sum())


'''
OUTPUT MODULE
'''
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
