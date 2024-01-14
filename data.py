import pandas as pd
import numpy as np


def preprocess_x(path_x):
    raw_x = pd.read_csv(path_x, low_memory=False)
    final_df = pd.DataFrame()

    patient_ids = raw_x['patientunitstayid'].unique()
    basic_features = raw_x[['admissionheight', 'admissionweight', 'age','ethnicity', 'unitvisitnumber']].iloc[:patient_ids.size]

    for f in ['admissionheight', 'admissionweight']:
        avg = basic_features[f].dropna().mean()
        basic_features[f].fillna(avg, inplace=True)
        final_df[f] = basic_features[f]

    basic_features['age'].replace('> 89', '90', inplace=True)
    avg = basic_features['age'].dropna().astype(int).mean()
    basic_features['age'].fillna(avg, inplace=True)
    final_df['age'] = basic_features['age'].astype(int)

    # Process ethnicity and visited
    basic_features['ethnicity'].fillna('Other/Unknown', inplace=True)
    final_df['ethnicity'] = basic_features['ethnicity']
    # print(df['ethnicity'])

    # Change unitvisitnumber nan to 1
    basic_features['unitvisitnumber'].fillna(1, inplace=True)
    final_df['unitvisitnumber'] = basic_features['unitvisitnumber']

    feature_db = [
              {'name': 'glucose', 'if_nan': 85, 'from': 'labname', 'col': 'labresult'},
              {'name': 'pH', 'if_nan': 7.4, 'from': 'labname', 'col': 'labresult'},
              {'name': 'Respiratory Rate', 'if_nan': 15, 'from': 'nursingchartcelltypevalname', 'col': 'nursingchartvalue'},
              {'name': 'O2 Saturation', 'if_nan': 98, 'from': 'nursingchartcelltypevalname', 'col': 'nursingchartvalue'},
              {'name': 'Heart Rate', 'if_nan': 60, 'from': 'nursingchartcelltypevalname', 'col': 'nursingchartvalue'},
              {'name': 'Non-Invasive BP Systolic', 'if_nan': 120, 'from': 'nursingchartcelltypevalname', 'col': 'nursingchartvalue'},
              {'name': 'Non-Invasive BP Diastolic', 'if_nan': 80, 'from': 'nursingchartcelltypevalname', 'col': 'nursingchartvalue'},
              {'name': 'Invasive BP Diastolic', 'if_nan': 80, 'from': 'nursingchartcelltypevalname', 'col': 'nursingchartvalue'},
              {'name': 'Invasive BP Systolic', 'if_nan': 120, 'from': 'nursingchartcelltypevalname', 'col': 'nursingchartvalue'},
              {'name': 'GCS Total', 'if_nan': 15, 'from': 'nursingchartcelltypevalname', 'col': 'nursingchartvalue'},
              {'name': 'Non-Invasive BP Mean', 'if_nan': 100, 'from': 'nursingchartcelltypevalname', 'col': 'nursingchartvalue'},
              {'name': 'Invasive BP Mean', 'if_nan': 100, 'from': 'nursingchartcelltypevalname', 'col': 'nursingchartvalue'}
              ]
    
    for feature in feature_db:
        new_col = []
        subset = raw_x[raw_x[feature['from']] == feature['name']][['patientunitstayid', feature['from'], feature['col']]]
        subset_mean = subset[feature['col']].replace('Unable to score due to medication', None).dropna().astype(float)
        mean = subset_mean.mean()
        for id in patient_ids:
            subset_id = subset[subset['patientunitstayid'] == id]
            try:
                if not subset_id.empty:
                    new_col.append(subset_id[feature['col']].astype(int).mean())
                else:
                    new_col.append(mean)
            except:
                new_col.append(feature['if_nan'])
        final_df[feature['name']] = new_col

    train_x = pd.get_dummies(final_df)

    train_x = train_x[['admissionheight', 'admissionweight', 'unitvisitnumber', 'age', 'glucose', 'pH', 'O2 Saturation', 'Heart Rate', 'GCS Total',
                 'ethnicity_African American', 'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic', 
                 'ethnicity_Other/Unknown',
                 'Non-Invasive BP Mean', 'Invasive BP Mean',
                 'Invasive BP Systolic', 'Invasive BP Diastolic',
                 'Non-Invasive BP Diastolic', 'Non-Invasive BP Systolic'
                 ]].to_numpy()

    return train_x

def preprocess_y(path_y):
    train_y = pd.Series(pd.read_csv(path_y)['hospitaldischargestatus']).to_numpy()
    return train_y

def returnCombined(path_x, pred_y):
    raw_test_x = pd.read_csv(path_x, low_memory=False)
    test_y_df = pd.DataFrame()
    test_y_df['patientunitstayid'] = raw_test_x['patientunitstayid'].unique()

    test_y_df['hospitaldischargestatus'] = pred_y
    return test_y_df

def saveDF(df: pd.DataFrame, filename="output.csv"):
    df['patientunitstayid'] = df['patientunitstayid'].astype(int)
    df['hospitaldischargestatus'] = df['hospitaldischargestatus'].astype(float)
    df.to_csv(filename, index=False)


