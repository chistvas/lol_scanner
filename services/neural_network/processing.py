import cv2
import os
import glob

# Настройки путей
INPUT_DIR = "dataset" 
OUTPUT_DIR = "processed_data"
KDA_DIR = os.path.join(OUTPUT_DIR, "kda_cs")

# КРИТИЧНО: Создаем папки, если их нет
os.makedirs(KDA_DIR, exist_ok=True)

def preprocess_zone(img):
    """Превращает цветной фрагмент в ЧБ"""
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Порог 200 может быть слишком жестким, если шрифт серый. 
    # Если не будет сохранять — попробуйте снизить до 150.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def process_screenshots():
    # Проверяем, есть ли файлы в папке
    images = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    if not images:
        print(f"Ошибка: В папке '{INPUT_DIR}' не найдено .jpg файлов!")
        return

    print(f"Найдено изображений: {len(images)}")

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None: 
            print(f"Не удалось прочитать: {img_path}")
            continue

        filename = os.path.basename(img_path)
        
        # Вырезаем зону KDA. 
        kda_zone = frame[0:30, 1650:1920] 

        # Обработка
        kda_clean = preprocess_zone(kda_zone)

        if kda_clean is not None:
            save_path = os.path.join(KDA_DIR, f"k_{filename}")
            success = cv2.imwrite(save_path, kda_clean)
            if success:
                print(f"Сохранено: {save_path}")
            else:
                print(f"Ошибка записи: {save_path}")

    print("--- Обработка завершена! ---")

if __name__ == "__main__":
    process_screenshots()