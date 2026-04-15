import cv2
import yt_dlp
import os
import random
import time

def download_frames(urls, output_dir="dataset", interval_sec=60, jitter_sec=15):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Настройки для извлечения прямой ссылки на видео без скачивания файла
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]/best[ext=mp4]',
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url_idx, url in enumerate(urls):
            print(f"Обработка видео: {url}")
            try:
                info = ydl.extract_info(url, download=False)
                video_url = info['url']
                video_title = info.get('title', f'video_{url_idx}')
            except Exception as e:
                print(f"Ошибка при получении ссылки {url}: {e}")
                continue

            cap = cv2.VideoCapture(video_url)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if not fps or fps == 0:
                fps = 30 # Резервное значение
            
            current_time_sec = 120  # Начинаем с 10-й секунды, чтобы пропустить заставки
            
            while True:
                # Рассчитываем следующий момент времени с учетом джиттера (разброса)
                # Например, 60 сек +- 15 сек
                actual_interval = interval_sec + random.uniform(-jitter_sec, jitter_sec)
                
                # Переходим к нужному кадру
                frame_to_seek = int(current_time_sec * fps)
                if frame_to_seek >= total_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_seek)
                ret, frame = cap.read()
                
                if not ret:
                    break

                # Сохраняем кадр
                timestamp_str = time.strftime('%H%M%S', time.gmtime(current_time_sec))
                filename = f"lol_snap_{url_idx}_{timestamp_str}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                
                print(f"Сохранен кадр: {filename} (время {current_time_sec:.1f}с)")
                
                current_time_sec += actual_interval

            cap.release()

# Пример использования
video_links = [
    "https://www.youtube.com/watch?v=MSAv87XaaeI"
]

download_frames(video_links, interval_sec=60, jitter_sec=15)