import os
import warnings
from src.loader import loading
from src.preprocessor import prep
from src.modeller import opti
from src.eval import eval

warnings.filterwarnings('ignore')

OUTPUT_DIR = "output"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Kalp Hastalığı Tahmini")

    print(" ")
    print("Veri")
    print(" ")

    X, y = loading()
    print(" ")
    print(f"Veri boyutu: {X.shape[0]} satır × {X.shape[1]} öznitelik\n")

    print("Veri Ön İşlemesi")
    print("\n")
    X_train, X_test, y_train, y_test = prep(X, y)
    print()

    print("GridSearchCV Hiperparametre Optimizasyonu")
    print(" ")
    models = opti(X_train, y_train)

    print("\nModel Değerlendirme, K-Fold Validation ve Grafikler")
    print(" ")
    results = eval(models, X_train, y_train, X_test, y_test)

    print("\n")
    print("Analiz tamam")
    print(f"Sonuçlar '{OUTPUT_DIR}/' klasöründe.")
    print(" ")

if __name__ == "__main__":
    main()