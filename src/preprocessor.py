import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def prep(X, y):
    print("Veri ön işleme")

    imputer = SimpleImputer(strategy='most_frequent')
    XImp= pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # One-Hot
    categorical = ['cp', 'restecg', 'slope', 'thal', 'exang', 'fbs', 'sex']
    categorical_encoding = []

    for col in categorical:
        if col in XImp.columns:
            categorical_encoding.append(col)

    XEncod = pd.get_dummies(XImp, columns=categorical_encoding, drop_first=True)

    # %80 eğitim, %20 test ayrılyor
    X_train, X_test, y_train, y_test = train_test_split(XEncod, y, test_size=0.2, random_state=42, stratify=y) # teste de aynı oranda gitsin diye)

    # StandardScaler — Z-skoru normalizasyonu
    scaler = StandardScaler()
    X_trainSca = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_testSca = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print(f"Eğitim: {X_trainSca.shape[0]} tane | Test: {X_testSca.shape[0]} tane")
    return X_trainSca, X_testSca, y_train, y_test