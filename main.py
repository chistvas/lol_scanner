import joblib
import numpy as np
from neural_network.main_data import analyze_league_screen
import os

# Определяем папку, в которой лежит сам запущенный скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Загружаем "Судью" (Random Forest)
rf_model = joblib.load('random_forest/rank_predictor_model.pkl')
label_encoder = joblib.load('random_forest/rank_label_encoder.pkl')

def get_final_verdict(img_path, role_id):
    if not os.path.exists(img_path):
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл {img_path} не найден!")
        return None, None
    
    stats = analyze_league_screen(img_path)
    
    # ПРОВЕРКА: если анализатор вернул None
    if stats is None:
        print("ОШИБКА: Анализатор скриншота не смог распознать данные.")
        return None, None
    
    # Теперь безопасно обращаемся к ключам
    features = np.array([[role_id, stats['time'], stats['kills'], ...]])
    
    # Шаг А: Распознаем текст через твою CNN
    # stats = { 'kills': 5, 'deaths': 2, 'assists': 10, 'time': 25.5, 'creeps': 180 }
    stats = analyze_league_screen(img_path) 
    
    # Шаг Б: Подготовка данных для Forest
    # Порядок должен быть как в обучении: [role, time_min, kills, deaths, assists, creeps]
    features = np.array([[
        role_id, 
        stats['time'], 
        stats['kills'], 
        stats['deaths'], 
        stats['assists'], 
        stats['creeps']
    ]])
    
    # Шаг В: Предсказание ранга
    prediction_idx = rf_model.predict(features)
    rank = label_encoder.inverse_transform(prediction_idx)[0]
    
    return rank, stats

# ТЕСТ
role_name = {0: "TOP", 1: "JNG", 2: "MID", 3: "ADC", 4: "SUP"}
my_role = 3 # Допустим, играем на ADC

final_rank, final_stats = get_final_verdict("neural_network/dataset/test.jpg", my_role)

print(f"Игра на позиции: {role_name[my_role]}")
print(f"Статистика: {final_stats['kills']}/{final_stats['deaths']}/{final_stats['assists']}")
print(f"ИТОГОВАЯ ОЦЕНКА: {final_rank}")