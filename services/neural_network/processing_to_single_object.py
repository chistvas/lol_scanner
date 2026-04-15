import cv2
import os
import glob
import numpy as np

# Настройки путей
INPUT_DIR = "processed_data/kda_cs"
OUTPUT_DIR = "dataset_chars"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def segment_with_colon_support(img_path):
    # 1. Загрузка изображения
    img = cv2.imread(img_path)
    if img is None:
        print(f"Ошибка загрузки: {img_path}")
        return

    # 2. Создание маски для поиска контуров
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Пороговая бинаризация (черный фон, белый текст)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # --- МОРФОЛОГИЯ ДЛЯ ДВOЕТОЧИЯ ---
    # Создаем вертикальное ядро, чтобы "склеить" верхнюю и нижнюю точки ':'
    # (3, 1) означает 3 пикселя в высоту и 1 в ширину
    kernel = np.ones((3, 1), np.uint8)
    # iterations=1 обычно достаточно, чтобы точки соприкоснулись
    detection_mask = cv2.dilate(binary, kernel, iterations=1)

    # 3. Поиск контуров на "раздутой" маске
    contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Порог высоты снижен до 6 пикселей специально для двоеточия
        if h > 6:
            char_boxes.append((x, y, w, h))

    # 4. Сортировка слева направо (чтобы символы шли по порядку)
    char_boxes.sort(key=lambda b: b[0])

    # 5. Сохранение результатов
    base_name = os.path.basename(img_path).split('.')[0]
    
    for i, (x, y, w, h) in enumerate(char_boxes):
        # Вырезаем из ОРИГИНАЛЬНОГО изображения (не из раздутой маски)
        char_img = img[y:y+h, x:x+w]
        
        # Стандартный размер для нейросети (32x32)
        target_h = 24
        scale = target_h / h
        new_w = int(w * scale)
        if new_w > 30: new_w = 30 # Ограничение по ширине
        
        # Ресайз с сохранением пропорций
        res_img = cv2.resize(char_img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Создаем пустой холст (черный квадрат)
        if len(img.shape) == 3:
            final_square = np.zeros((32, 32, 3), dtype=np.uint8)
            y_off = (32 - target_h) // 2
            x_off = (32 - new_w) // 2
            final_square[y_off:y_off+target_h, x_off:x_off+new_w, :] = res_img
        else:
            final_square = np.zeros((32, 32), dtype=np.uint8)
            y_off = (32 - target_h) // 2
            x_off = (32 - new_w) // 2
            final_square[y_off:y_off+target_h, x_off:x_off+new_w] = res_img

        # Сохраняем
        output_filename = f"{base_name}_{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, output_filename), final_square)

if __name__ == "__main__":
    # Ищем все jpg файлы в папке
    files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    print(f"Начинаю обработку {len(files)} файлов...")
    
    for f in files:
        segment_with_colon_support(f)
        
    print(f"Готово! Проверь папку: {OUTPUT_DIR}")