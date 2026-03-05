import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_database(df):
    cols_to_fix=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in cols_to_fix:
        # 1. Mantıksız '0' değerlerini NaN ile değiştiriyoruz
        df[col]=df[col].replace(0,np.nan)

    for col in cols_to_fix:
        # 2. Eksik verileri (NaN) o sütunun ortalaması veya medyanı ile dolduruyoruz
        df[col]=df[col].fillna(df[col].median())

    return df

def scale_data(df):
    scaler=MinMaxScaler()
    features=df.drop('Outcome',axis=1)
    
    scaled_features=scaler.fit_transform(features)

    return scaled_features,df['Outcome'],scaler
