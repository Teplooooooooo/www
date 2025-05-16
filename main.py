#https://git.altlinux.org/beehive/logs/Sisyphus/x86_64/latest/error/aces_container-1.0.2-alt1_8
import requests

# Замените на нужный эндпоинт или параметры
url = "https://git.altlinux.org/beehive/logs/Sisyphus/x86_64/latest/error/aces_container-1.0.2-alt1_8"

response = requests.get(url)
response.raise_for_status()  # Проверка на ошибки

# Если API возвращает JSON
try:
    data = response.json()
    print("Полученные данные (JSON):")
    print(data)
except ValueError:
    # Если возвращается просто текст
    print("Полученный текст:")
    print(response.text)
