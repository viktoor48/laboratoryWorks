from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

# Шаг 3: Генерация данных с использованием make_classification
X, y = make_classification(
    n_samples=100,  # Количество образцов
    n_features=2,   # Количество признаков
    n_redundant=0,   # Количество избыточных признаков
    n_informative=1, # Количество информативных признаков
    n_clusters_per_class=1,  # Количество кластеров на класс
    class_sep=0.45,  # Разделение классов
    random_state=78  # Random State для воспроизводимости
)

# Шаг 4: Вывести первые 15 элементов выборки
data = pd.DataFrame({'Feature 1': X[:, 0], 'Feature 2': X[:, 1], 'Label': y})
print(data.head(15))
#Шаг 5: Отобразить на графике сгенерированную выборку
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Classification Data')
plt.show()
# Шаг 6: Разбить данные на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Шаг 7: Отобразить обучающую и тестовую выборки
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='s', label='Test')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Train and Test Data')
plt.legend()
plt.show()
# Шаг 8: Реализовать и обучить модели классификаторов
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
import numpy as np

# Функция для вывода результатов классификации
def print_classification_results(y_true, y_pred, model_name):
    print(f"Results for {model_name}:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred)}")
    print(f"Recall: {recall_score(y_true, y_pred)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"AUC-ROC Score: {roc_auc_score(y_true, y_pred)}\n")

# a) Метод к-ближайших соседей
k_values = [1, 3, 5, 9]

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    print_classification_results(y_test, knn_pred, f"K-Nearest Neighbors (k={k})")

# b) Наивный байесовский метод
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print_classification_results(y_test, nb_pred, "Naive Bayes")

# c) Случайный лес
n_estimators_values = [5, 10, 15, 20, 50]

for n_estimators in n_estimators_values:
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print_classification_results(y_test, rf_pred, f"Random Forest (n_estimators={n_estimators})")

# Отобразить область принятия решений для k-ближайших соседей с k=5
knn_model_5 = KNeighborsClassifier(n_neighbors=5)
knn_model_5.fit(X_train, y_train)
plot_decision_regions(X, y, knn_model_5, "K-Nearest Neighbors Decision Regions (k=5)")# Шаг 8: Реализовать и обучить модели классификаторов
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
import numpy as np

# Функция для вывода результатов классификации
def print_classification_results(y_true, y_pred, model_name):
    print(f"Results for {model_name}:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred)}")
    print(f"Recall: {recall_score(y_true, y_pred)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"AUC-ROC Score: {roc_auc_score(y_true, y_pred)}\n")

# a) Метод к-ближайших соседей
k_values = [1, 3, 5, 9]

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    print_classification_results(y_test, knn_pred, f"K-Nearest Neighbors (k={k})")

# b) Наивный байесовский метод
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print_classification_results(y_test, nb_pred, "Naive Bayes")

# c) Случайный лес
n_estimators_values = [5, 10, 15, 20, 50]

for n_estimators in n_estimators_values:
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print_classification_results(y_test, rf_pred, f"Random Forest (n_estimators={n_estimators})")

# Отобразить область принятия решений для k-ближайших соседей с k=5
knn_model_5 = KNeighborsClassifier(n_neighbors=5)
knn_model_5.fit(X_train, y_train)
plot_decision_regions(X, y, knn_model_5, "K-Nearest Neighbors Decision Regions (k=5)")
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Генерация данных
X, y = make_classification(n_samples=500, n_features=2, n_informative=1, n_redundant=0,
                           n_clusters_per_class=1, class_sep=0.45, random_state=78)

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Отобразить на графике обучающую и тестовую выборки
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x', s=80, label='Test')
plt.title('Train and Test Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# a) Метод к-ближайших соседей
k_values = [1, 3, 5, 9]

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    print_classification_results(y_test, knn_pred, f"K-Nearest Neighbors (k={k})")

# b) Наивный байесовский метод
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print_classification_results(y_test, nb_pred, "Naive Bayes")

# c) Случайный лес
n_estimators_values = [5, 10, 15, 20, 50]

for n_estimators in n_estimators_values:
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print_classification_results(y_test, rf_pred, f"Random Forest (n_estimators={n_estimators})")
# Разбиение данных на обучающую и тестовую выборки (10% тестовая)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Реализация, обучение и тестирование моделей
def classify_and_evaluate(classifier, model_name, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Вывод результатов
    print(f"\nResults for {model_name} with 10% test set:")
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print(f"AUC ROC: {metrics.roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])}")

# a) k-ближайших соседей (n_neighbors = {1, 3, 5, 9})
for n_neighbors in [1, 3, 5, 9]:
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classify_and_evaluate(knn_classifier, f"KNN (k={n_neighbors})", X_train, X_test, y_train, y_test)

# b) Наивный байесовский метод
nb_classifier = GaussianNB()
classify_and_evaluate(nb_classifier, "Naive Bayes", X_train, X_test, y_train, y_test)

# c) Случайный лес (n_estimators = {5, 10, 15, 20, 50})
for n_estimators in [5, 10, 15, 20, 50]:
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    classify_and_evaluate(rf_classifier, f"Random Forest (n={n_estimators})", X_train, X_test, y_train, y_test)
# Разбиение данных на обучающую и тестовую выборки (35% тестовая)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Повторение процесса с новыми тестовыми данными
# a) k-ближайших соседей (n_neighbors = {1, 3, 5, 9})
for n_neighbors in [1, 3, 5, 9]:
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classify_and_evaluate(knn_classifier, f"KNN (k={n_neighbors})", X_train, X_test, y_train, y_test)

# b) Наивный байесовский метод
nb_classifier = GaussianNB()
classify_and_evaluate(nb_classifier, "Naive Bayes", X_train, X_test, y_train, y_test)

# c) Случайный лес (n_estimators = {5, 10, 15, 20, 50})
for n_estimators in [5, 10, 15, 20, 50]:
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    classify_and_evaluate(rf_classifier, f"Random Forest (n={n_estimators})", X_train, X_test, y_train, y_test)
import pandas as pd

# Создание DataFrame для хранения результатов
results_df = pd.DataFrame(columns=['Model', 'Test Size', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'])

# Реализация, обучение и тестирование моделей
def classify_and_evaluate(classifier, model_name, X_train, X_test, y_train, y_test, test_size):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Вычисление метрик
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    auc_roc = metrics.roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])

    # Добавление результатов в DataFrame
    results_df.loc[len(results_df)] = [model_name, test_size, accuracy, precision, recall, f1_score, auc_roc]

# Разбиение данных на обучающую и тестовую выборки (10% тестовая)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# a) k-ближайших соседей (n_neighbors = {1, 3, 5, 9})
for n_neighbors in [1, 3, 5, 9]:
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classify_and_evaluate(knn_classifier, f"KNN (k={n_neighbors})", X_train, X_test, y_train, y_test, 0.1)

# b) Наивный байесовский метод
nb_classifier = GaussianNB()
classify_and_evaluate(nb_classifier, "Naive Bayes", X_train, X_test, y_train, y_test, 0.1)

# c) Случайный лес (n_estimators = {5, 10, 15, 20, 50})
for n_estimators in [5, 10, 15, 20, 50]:
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    classify_and_evaluate(rf_classifier, f"Random Forest (n={n_estimators})", X_train, X_test, y_train, y_test, 0.1)

# Разбиение данных на обучающую и тестовую выборки (35% тестовая)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Разбиение данных на обучающую и тестовую выборки (35% тестовая)
X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(X, y, test_size=0.35, random_state=42)

# a) k-ближайших соседей (n_neighbors = {1, 3, 5, 9})
for n_neighbors in [1, 3, 5, 9]:
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classify_and_evaluate(knn_classifier, f"KNN (k={n_neighbors})", X_train_large, X_test_large, y_train_large, y_test_large, 0.35)

# b) Наивный байесовский метод
nb_classifier = GaussianNB()
classify_and_evaluate(nb_classifier, "Naive Bayes", X_train_large, X_test_large, y_train_large, y_test_large, 0.35)

# c) Случайный лес (n_estimators = {5, 10, 15, 20, 50})
for n_estimators in [5, 10, 15, 20, 50]:
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    classify_and_evaluate(rf_classifier, f"Random Forest (n={n_estimators})", X_train_large, X_test_large, y_train_large, y_test_large, 0.35)

# Вывод результатов
print(results_df)