import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# --- 1. Настройки ---
BATCH_SIZE = 8      # Маленький батч, так как данных мало
EPOCHS = 15          # Больше эпох, чтобы модель успела выучить на малом объеме
LEARNING_RATE = 0.001
DATA_DIR = "train_data" # Папка с твоим мини-датасетом
MODEL_SAVE_PATH = "draft_model.pth"

# Определение устройства (GPU если есть, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Обучение на: {device}")

# --- 2. Подготовка данных ---
# Трансформы: переводим в тензор и нормализуем
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # На всякий случай в ЧБ
    transforms.Resize((32, 32)),                 # Принудительный ресайз
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))        # Нормализация ([-1, 1])
])

# Загрузчик данных из папок
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Сохраняем маппинг индексов на имена классов (важно для сортировки!)
class_names = train_dataset.classes
print(f"Классы: {class_names}")

# --- 3. Архитектура нейросети (SimpleCNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Сверточные слои: извлекают признаки (линии, круги)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Слои пулинга: уменьшают размер картинки
        self.pool = nn.MaxPool2d(2, 2)
        # Полносвязные слои: классифицируют признаки
        self.fc1 = nn.Linear(32 * 8 * 8, 128) # 32 канала * 8x8 пикселей после пулингов
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x -> Conv1 -> ReLU -> Pool (32x32 -> 16x16)
        x = self.pool(self.relu(self.conv1(x)))
        # x -> Conv2 -> ReLU -> Pool (16x16 -> 8x8)
        x = self.pool(self.relu(self.conv2(x)))
        # Сглаживание в вектор
        x = x.view(-1, 32 * 8 * 8)
        # ReLU -> FC2 (Выход)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Создаем модель
model = SimpleCNN(num_classes=len(class_names)).to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. Цикл обучения ---
print("Начало обучения...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()
        
        # Статистика
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_acc = 100 * correct / total
    print(f"Эпоха [{epoch+1}/{EPOCHS}], Потери: {running_loss/len(train_loader):.4f}, Точность: {epoch_acc:.2f}%")

# --- 5. Сохранение модели ---
# Критично: сохраняем модель И маппинг классов
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, MODEL_SAVE_PATH)
print(f"Черновая модель сохранена в {MODEL_SAVE_PATH}")