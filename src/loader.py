import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import os

def loading(csv_path='data/heart_disease.csv'):
    try:
        print("API üzerinden veri çekilmeye çalışılıyor")
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets
        df = pd.concat([X, y], axis=1)
        print("API üzerinden veri çekme tamam.")
    except Exception as e:
        print(f"API'ye ulaşılamadı: {e}")
        print("CSV den okunuyor")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError("CSV dosyası yok")

    # Hatalı Object tiplerini ('?')' den NaN yapma işi
    for col in ['ca', 'thal']:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].replace('?', np.nan), errors='coerce')
            
            df[col] = df[col].fillna(df[col].mode()[0])

    # Target sütunu binarye çeviriyorum (0 = sağlıklı, 1 = hasta)
    if 'diagnosis' in df.columns:
        targets = 'diagnosis'
    else:
        targets = 'num'

    target_list = []

    for val in df[targets]:
        if val > 0:
            target_list.append(1)
        else:
            target_list.append(0)

    df['target'] = target_list

    y = df['target']
    X = df.drop(columns=[targets, 'target'], errors='ignore')

    print(f"Veri seti yüklendi: {X.shape[0]} satır, {X.shape[1]} öznitelik.")
    return X, y