import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Загрузка данных
try:
    df = pd.read_csv("score_data_generator/league_expert_data.csv")
    print("Данные успешно загружены!")
except FileNotFoundError:
    print("Ошибка: Файл 'league_expert_data.csv' не найден. Сначала запусти генератор.")
    exit()

# 2. Подготовка признаков (X) и целевой переменной (y)
# Мы убираем колонку 'rank', так как это то, что мы хотим предсказать
X = df.drop('rank', axis=1)
y = df['rank']

# Нейросети и леса лучше работают с числами, поэтому переведем S, A, B в 0, 1, 2...
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Разделение на обучающую и тестовую выборки (80% учимся, 20% проверяем)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Создание и обучение модели
# n_estimators=100 — количество деревьев в "лесу"
# max_depth=10 — ограничиваем глубину, чтобы модель не "зубрила" данные (overfitting)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
print("Начинаю обучение Random Forest...")
rf_model.fit(X_train, y_train)

# 5. Оценка качества
accuracy = rf_model.score(X_test, y_test)
print(f"\nТочность модели на тестовых данных: {accuracy:.2%}")

# Выведем детальный отчет (precision/recall для каждого ранга)
y_pred = rf_model.predict(X_test)
print("\nДетальный отчет по рангам:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 6. Анализ важности признаков (Feature Importance)
# Это покажет, на что модель смотрит в первую очередь
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Параметр': feature_names, 'Важность': importances})
print("\nЧто больше всего влияет на оценку:")
print(feature_importance_df.sort_values(by='Важность', ascending=False))

# 7. СОХРАНЕНИЕ МОДЕЛИ
# Сохраняем модель и энкодер, чтобы использовать их в main.py без переобучения
joblib.dump(rf_model, 'random_forest/rank_predictor_model.pkl')
joblib.dump(label_encoder, 'random_forest/rank_label_encoder.pkl')
print("\nМодель сохранена как 'random_forest/rank_predictor_model.pkl'")