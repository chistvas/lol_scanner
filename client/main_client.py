import sys
import os
import time
import random
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal
import pygetwindow as gw
import pyautogui

# Класс для работы в фоновом режиме
class ScreenshotThread(QThread):
    log_signal = pyqtSignal(str)  # Сигнал для передачи логов в интерфейс

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        self.running = True
        target_title = "League of Legends (TM) Client" # Точное название окна игры
        
        if not os.path.exists("lol_screenshots"):
            os.makedirs("lol_screenshots")

        while self.running:
            # Ищем окно игры
            windows = gw.getWindowsWithTitle(target_title)
            
            # Проверяем, что окно существует и оно сейчас активно (на переднем плане)
            # Если нужно снимать даже когда игра свернута, уберите .isActive
            active_window = next((w for w in windows if target_title in w.title), None)

            if active_window:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"lol_screenshots/screen_{timestamp}.png"
                
                try:
                    # Делаем скриншот только области окна игры
                    screenshot = pyautogui.screenshot(region=(
                        active_window.left, 
                        active_window.top, 
                        active_window.width, 
                        active_window.height
                    ))
                    screenshot.save(filename)
                    self.log_signal.emit(f"📸 Снято: {timestamp}")
                except Exception as e:
                    self.log_signal.emit(f"❌ Ошибка захвата: {e}")
            else:
                self.log_signal.emit("💤 Игра не запущена. Жду...")

            # Рандомный интервал 60 +/- 15 сек
            wait_time = random.randint(45, 75)
            for _ in range(wait_time):
                if not self.running: break
                time.sleep(1)

    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screenshot Prototype")
        self.resize(400, 300)

        # Интерфейс
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Статус: Ожидание")
        layout.addWidget(self.status_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        self.btn_toggle = QPushButton("Запустить")
        self.btn_toggle.clicked.connect(self.handle_button)
        layout.addWidget(self.btn_toggle)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Поток
        self.worker = ScreenshotThread()
        self.worker.log_signal.connect(self.update_log)

    def update_log(self, text):
        self.log_view.append(text)

    def handle_button(self):
        if not self.worker.isRunning():
            self.worker.start()
            self.btn_toggle.setText("Остановить")
            self.status_label.setText("Статус: Работает")
            self.update_log("🚀 Мониторинг запущен...")
        else:
            self.worker.stop()
            self.btn_toggle.setText("Запустить")
            self.status_label.setText("Статус: Остановлен")
            self.update_log("🛑 Мониторинг остановлен.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())