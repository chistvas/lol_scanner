import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os


# --- 1. АРХИТЕКТУРА НЕЙРОСЕТИ (Должна совпадать с cnn_mini.py) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. НАСТРОЙКИ И ЗАГРУЗКА ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "neural_network","train_mini", "draft_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Файл модели {MODEL_PATH} не найден!")
    exit()

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']
model = SimpleCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- 3. ФУНКЦИИ ОБРАБОТКИ ---

def get_string_from_zone(zone_img, is_time=False):
    """Нарезает зону на символы и распознает их, игнорируя мусор"""
    if zone_img is None or zone_img.size == 0: 
        return ""
    
    # Подготовка маски
    gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    
    # Морфология (склейка двоеточия)
    kernel = np.ones((3, 1), np.uint8)
    mask = cv2.dilate(binary, kernel, iterations=1)
    
    # Поиск символов
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[3] > 6]
    boxes.sort(key=lambda b: b[0])
    
    res_str = ""
    for (x, y, w, h) in boxes:
        # Вырезаем символ и готовим 32x32
        char_crop = binary[y:y+h, x:x+w]
        final_sq = np.zeros((32, 32), dtype=np.uint8)
        
        target_h = 24
        target_w = int(w * (target_h / h))
        target_w = min(target_w, 30)
        
        char_res = cv2.resize(char_crop, (target_w, target_h))
        y_off, x_off = (32 - target_h) // 2, (32 - target_w) // 2
        final_sq[y_off:y_off+target_h, x_off:x_off+target_w] = char_res
        
        # Предсказание нейросетью
        pil_img = Image.fromarray(final_sq)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            cls = class_names[pred.item()]
            
            # --- ФИЛЬТРАЦИЯ И МАППИНГ ---
            if cls == 'garbage':
                continue
            elif cls == 'slash':
                res_str += '/'
            elif cls == 'colon':
                res_str += ':'
            else:
                res_str += cls # Если это цифра "0"-"9"
                
    return res_str

def analyze_league_screen(img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Ошибка: Не удалось открыть {img_path}")
        return

    # --- КООРДИНАТЫ (Подправь под свой интерфейс!) ---
    # kda_zone = frame[Y1:Y2, X1:X2]
    kda_zone  = frame[1:35, 1650:1740]
    time_zone = frame[1:35, 1850:1915]
    cs_zone   = frame[1:35, 1775:1830]

    # Визуальная отладка (раскомментируй, чтобы проверить точность кропа)
    cv2.imwrite("debug_kda.jpg", kda_zone)
    cv2.imwrite("debug_time.jpg", time_zone)
    cv2.imwrite("debug_cs.jpg", cs_zone)

    # 1. Распознавание строк
    kda_raw  = get_string_from_zone(kda_zone)
    time_raw = get_string_from_zone(time_zone, is_time=True)
    cs_raw   = get_string_from_zone(cs_zone)

    # 2. Парсинг KDA
    parts = kda_raw.split('/')
    k = parts[0] if len(parts) > 0 else "0"
    d = parts[1] if len(parts) > 1 else "0"
    a = parts[2] if len(parts) > 2 else "0"

    # 3. Парсинг Времени (умная вставка двоеточия)
    time_clean = time_raw.replace(':', '')
    if len(time_clean) >= 3:
        t_min, t_sec = time_clean[:-2], time_clean[-2:]
    else:
        t_min, t_sec = "00", "00"

    # 4. Парсинг Крипов
    creeps = "".join(filter(str.isdigit, cs_raw))

    # ВЫВОД РЕЗУЛЬТАТОВ
    print("\n" + "="*25)
    print(f"СТАТИСТИКА МАТЧА:")
    print(f"Kills   : {k}")
    print(f"Deaths  : {d}")
    print(f"Assists : {a}")
    print(f"Time    : {t_min}:{t_sec}")
    print(f"Creeps  : {creeps}")
    print("="*25)

    # Создаем словарь с результатами
    results = {
        'kills': int(k) if k.isdigit() else 0,
        'deaths': int(d) if d.isdigit() else 0,
        'assists': int(a) if a.isdigit() else 0,
        'time': float(t_min) + float(t_sec)/60, # Переводим время в число для леса
        'creeps': int(creeps) if creeps.isdigit() else 0
    }
    
    # ВАЖНО: эта строчка должна быть!
    return results

if __name__ == "__main__":
    # Укажи имя своего файла со скриншотом
    analyze_league_screen("dataset/test5.jpg")