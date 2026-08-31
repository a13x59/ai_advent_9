import os
import requests

# Получаем путь к папке, где находится скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))
secrets_path = os.path.join(script_dir, "secrets")

# Читаем API-ключ из файла
try:
    with open(secrets_path, "r") as f:
        API_KEY = f.read().strip()
except FileNotFoundError:
    print("Файл 'secrets' не найден в папке со скриптом.")
    exit(1)

url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

data = {
    "model": "deepseek-chat",
    # "model": "deepseek-reasoner",
    "messages": [
        {"role": "user", "content": "Чем отилчается ДВС от электродвигателя? Ответ выведи в виде сравнительной таблицы"}
        # {"role": "user", "content": "Как называется дебютный альбом группы Nirvana?"}
    ],
    "stream": False
    # "stream": True # TODO
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    answer = result["choices"][0]["message"]["content"]
    print("Ответ:", answer)
else:
    print(f"Ошибка {response.status_code}: {response.text}")
