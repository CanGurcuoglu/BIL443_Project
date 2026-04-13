import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score

OUTPUT_DIR = "output"

def eval(models, X_train, y_train, X_test, y_test):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    results = []
    rocs = {}

    for n in models:
        model = models[n]

        print("\nModel =>:", n)

        # Cross Validation
        scores = cross_val_score(model, X_train, y_train, cv=5)
        cvMean = scores.mean() * 100
        cvSTD = scores.std() * 100

        print("CV Accuracy: %.2f (+- %.2f)" % (cvMean, cvSTD))

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accur = accuracy_score(y_test, y_pred) * 100
        print("Test Accuracy =>:", accur)

        # Classification
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(cm)

        # Metrics
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall) 
        fnr = fn / (tp + fn)
        specificity = tn / (tn + fp)

        # ROC
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, p)
            roc_auc = auc(fpr, tpr)
            rocs[n] = (fpr, tpr, roc_auc)

        # Confusion matrix
        plt.figure()
        plt.imshow(cm, cmap="Blues")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j])

        plt.title("Confusion Matrix - " + n)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.colorbar()

        plt.savefig(os.path.join(OUTPUT_DIR, n + "_confmatrix.png"))
        plt.close()

        results.append({"Model": n,"CV Accuracy (%)": cvMean,"CV Std (%)": cvSTD,"Test Accuracy (%)": accur,"Recall (%)": recall * 100,"Precision (%)": precision * 100,"F1 (%)": f1 * 100,"FN Rate (%)": fnr * 100,"Specificity (%)": specificity * 100})
    # ROC
    
    plt.figure()

    for n in rocs:
        fpr, tpr, roc_auc = rocs[n]
        plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curves")

    plt.savefig(os.path.join(OUTPUT_DIR, "roc.png"))
    plt.close()

    df = pd.DataFrame(results)

    df = df.sort_values(by="Test Accuracy (%)", ascending=False)

    print(df)

    df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)

    # ACCURACY
    
    plt.figure()
    plt.bar(df["Model"], df["Test Accuracy (%)"])
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=32)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison_graf.png"))
    plt.close()

    # FALSE NEGATIVE
    
    plt.figure()

    fnr_vals = df["FN Rate (%)"]

    plt.bar(df["Model"], fnr_vals, color="red")
    plt.title("Yanlış Negatif Oranı Karşılaştırması")
    plt.ylabel("FN (%)")
    plt.xticks(rotation=32)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "false_negative_rate_comparison_graph.png"))
    plt.close()

    # PRECISION / RECALL / F1 / ACCURACY

    x = np.arange(len(df["Model"]))
    width = 0.2

    plt.figure()

    plt.bar(x - 0.3, df["Test Accuracy (%)"], width, label="Accuracy")
    plt.bar(x - 0.1, df["Precision (%)"], width, label="Precision")
    plt.bar(x + 0.1, df["Recall (%)"], width, label="Recall")
    plt.bar(x + 0.3, df["F1 (%)"], width, label="F1")

    plt.xticks(x, df["Model"], rotation=32)
    plt.title("Performance Comparison")
    plt.ylabel("Score (%)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "metricsComparison_graph.png"))
    plt.close()

    print("\nGrafikler=>:", OUTPUT_DIR)

    return df