import sys
import os
import time
import random
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal

# Класс для работы в фоновом режиме
class ScreenshotThread(QThread):
    log_signal = pyqtSignal(str)  # Сигнал для передачи логов в интерфейс

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        self.running = True
        if not os.path.exists("shots"):
            os.makedirs("shots")

        while self.running:
            # Делаем скриншот
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"shots/screen_{timestamp}.png"
            
            try:
                import pyautogui
                pyautogui.screenshot(filename)
                self.log_signal.emit(f"✅ Сохранено: {filename}")
            except Exception as e:
                self.log_signal.emit(f"❌ Ошибка: {e}")

            # Вычисляем интервал: 60 сек +- 15 сек (от 45 до 75)
            wait_time = random.randint(45, 75)
            self.log_signal.emit(f"⏳ Следующий через {wait_time} сек...")
            
            # Спим короткими интервалами, чтобы можно было быстро остановить поток
            for _ in range(wait_time):
                if not self.running:
                    break
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