import pandas as pd
import numpy as np

def generate_league_data_with_roles(n_samples=3000):
    np.random.seed(42)
    
    # Список ролей: 0:Top, 1:Jungle, 2:Mid, 3:ADC, 4:Support
    roles = np.random.randint(0, 5, n_samples)
    
    data = []
    for role in roles:
        time_min = np.random.uniform(20, 40)
        
        # Базовые параметры зависят от роли
        if role == 1: # Jungle
            kills = np.random.randint(2, 12)
            deaths = np.random.randint(0, 8)
            assists = np.random.randint(4, 15)
            creeps = int(time_min * np.random.uniform(4, 7)) # Лесники фармят меньше лайнеров
        elif role == 4: # Support
            kills = np.random.randint(0, 4)
            deaths = np.random.randint(0, 10)
            assists = np.random.randint(10, 30)
            creeps = int(time_min * np.random.uniform(0, 2)) # Почти не фармят
        else: # Top, Mid, ADC
            kills = np.random.randint(3, 15)
            deaths = np.random.randint(0, 9)
            assists = np.random.randint(2, 10)
            creeps = int(time_min * np.random.uniform(7, 10)) # Высокий фарм
            
        # Рассчитываем "скрытый рейтинг" (Internal Score) для разметки ранга
        # Логика оценки меняется в зависимости от роли!
        kda_ratio = (kills + assists * 0.5) / (deaths + 1)
        cspm = creeps / time_min
        
        if role == 4: # Для саппорта важен KDA (ассисты), а не крипы
            internal_score = (kda_ratio * 20) + (assists * 2) - (deaths * 3)
        else: # Для остальных важны и KDA, и крипы
            internal_score = (kda_ratio * 12) + (cspm * 18) - (deaths * 2)

# Обновленные пороги для более жесткой оценки (балансировка классов)
        if internal_score > 200: rank = 'S'   # Было 130
        elif internal_score > 150: rank = 'A' # Было 90
        elif internal_score > 100: rank = 'B' # Было 60
        elif internal_score > 60: rank = 'C'  # Было 35
        else: rank = 'D'
        
        data.append([role, time_min, kills, deaths, assists, creeps, rank])

    columns = ['role', 'time_min', 'kills', 'deaths', 'assists', 'creeps', 'rank']
    df = pd.DataFrame(data, columns=columns)
    return df

# Генерация и сохранение
df_league = generate_league_data_with_roles(5000)
df_league.to_csv("league_expert_data.csv", index=False)

print("Пример сгенерированных данных:")
print(df_league.head(10))

# Проверка баланса классов
print("\nРаспределение рангов в датасете:")
print(df_league['rank'].value_counts())