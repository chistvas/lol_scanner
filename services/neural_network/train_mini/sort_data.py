import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import shutil
import glob

# --- 1. ТА ЖЕ САМАЯ АРХИТЕКТУРА (Обязательно!) ---
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

# --- 2. НАСТРОЙКИ ПУТЕЙ ---
MODEL_PATH = "draft_model.pth"
SOURCE_DIR = "../dataset_chars"         # Откуда берем (твои 2000 символов)
TARGET_DIR = "labeled_dataset_auto"  # Куда складываем
CONFIDENCE_THRESHOLD = 0.90          # Порог уверенности (90%)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. ЗАГРУЗКА МОДЕЛИ ---
checkpoint = torch.load(MODEL_PATH)
class_names = checkpoint['class_names']
model = SimpleCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Те же трансформации, что и при обучении
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Создаем папки для классов + папку для "сомнительных"
for cls in class_names:
    os.makedirs(os.path.join(TARGET_DIR, cls), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "unsure"), exist_ok=True)

# --- 4. ПРОЦЕСС СОРТИРОВКИ ---
files = glob.glob(os.path.join(SOURCE_DIR, "*.jpg"))
print(f"Найдено файлов для сортировки: {len(files)}")

with torch.no_grad():
    for f_path in files:
        try:
            filename = os.path.basename(f_path)
            img = Image.open(f_path)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Предсказание
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob_max, predicted_idx = torch.max(probabilities, 1)
            
            label = class_names[predicted_idx.item()]
            conf = prob_max.item()
            
            # Если уверены — в папку класса, если нет — в unsure
            if conf >= CONFIDENCE_THRESHOLD:
                shutil.copy(f_path, os.path.join(TARGET_DIR, label, filename))
            else:
                shutil.copy(f_path, os.path.join(TARGET_DIR, "unsure", filename))
                
        except Exception as e:
            print(f"Ошибка с файлом {f_path}: {e}")

print(f"Готово! Результаты в папке: {TARGET_DIR}")