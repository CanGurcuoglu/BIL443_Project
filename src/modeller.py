from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def opti(X_train, y_train):

    # K-NN - k optimize
    knnP = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'metric': ['euclidean', 'manhattan'],
        'weights': ['uniform', 'distance']
    }
    knnG = GridSearchCV(KNeighborsClassifier(), knnP, cv=5, scoring='accuracy')
    knnG.fit(X_train, y_train)

    print(f"k-NN => en iyi parametreler : {knnG.best_params_}")
    print(f"CV Accuracy: %{knnG.best_score_ * 100:.2f}\n")

    # Naive Bayes — var_smoothing => varyansı çok küçükse bölme işlemlerinde hata olabilir diyeek varyansa küçük bir değer ekleniyormuş
    nbP = {'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]}
    nbG = GridSearchCV(GaussianNB(), nbP, cv=5, scoring='accuracy')
    nbG.fit(X_train, y_train)
    print(f"Naive Bayes => en iyi parametreler : {nbG.best_params_}")
    print(f"CV Accuracy: %{nbG.best_score_ * 100:.2f}\n")

    # Karar Ağacı — derinlik ve bölme kriterleri 
    dtP = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],'criterion': ['gini', 'entropy']}
    dtG = GridSearchCV(DecisionTreeClassifier(random_state=42), dtP, cv=5, scoring='accuracy')
    dtG.fit(X_train, y_train)
    print(f"Karar Ağacı => en iyi parametreler : {dtG.best_params_}")
    print(f"CV Accuracy: %{dtG.best_score_ * 100:.2f}\n")

    # SVM — C, kernel ve gamma optimize
    svmP = {'C': [0.1, 1, 10, 100],'kernel': ['linear', 'rbf'],'gamma': ['scale', 'auto']} # => veri noktasının etki alanı
    
    svmG = GridSearchCV(SVC(probability=True, random_state=42), svmP, cv=5, scoring='accuracy')
    svmG.fit(X_train, y_train)
    print(f"SVM => en iyi parametreler : {svmG.best_params_}")
    print(f"CV Accuracy: %{svmG.best_score_ * 100:.2f}\n")

    # Lojistik Regresyon — C ve solver optimize ediliyor
    logRP = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [500, 1000]
    }
    logRPG = GridSearchCV(LogisticRegression(random_state=42), logRP, cv=5, scoring='accuracy')
    logRPG.fit(X_train, y_train)
    print(f"Lojistik Regresyon => en iyi parametreler : {logRPG.best_params_}")
    print(f"CV Accuracy: %{logRPG.best_score_ * 100:.2f}\n")

    bests = {
        "k-NN": knnG.best_estimator_,
        "Naive Bayes": nbG.best_estimator_,
        "Karar Agaclari": dtG.best_estimator_,
        "SVM": svmG.best_estimator_,
        "Lojistik Regresyon": logRPG.best_estimator_,
    }

    print(" ")
    print("GridSearchCV tamam")
    print(" ")
    return bests