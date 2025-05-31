import time
import sys

# mic_array kütüphanesinin yüklü olduğu klasörün yolunu ekle
sys.path.append("C:/Users/ahmet/mic_array")

from mic_array import MicArray

# 4 kanal kullanarak cihazı seç ve 4 kanal algoritmasını çalıştır
with MicArray(rate=16000, channels=4, chunk_size=16000//4) as mic:
    print("Konuşun; açı değerleri aşağıda akacak (Ctrl+C ile çık).")
    for frames in mic.read_chunks():
        angle = mic.get_direction(frames)
        if angle is not None:
            print(f"Açı: {angle:6.1f}°")
        time.sleep(1)
