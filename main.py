import pyaudio
import audioop
import time
import subprocess
import speech_recognition as sr
import wave
import re
import os
import keyboard
import locale
import pyttsx3
import winsound
import json
import hashlib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from g4f.client import Client  # Импортируем библиотеку g4f для работы с API

# Настройки
CHUNK = 1024  # Количество сэмплов за буфер
FORMAT = pyaudio.paInt16  # Формат аудио
CHANNELS = 1  # Каналы (моно)
RATE = 16000  # Частота дискретизации
THRESHOLD_START = 1000  # Порог громкости активации и завершения
THRESHOLD_CONTINUE = 50 # Порог громкости записи
SILENCE_DURATION = 1  # Время тишины для остановки записи (в секундах)
MIN_AUDIO_LENGTH = 2  # Минимальная длина записи в секундах
MAX_HISTORY_LENGTH = 10  # Настраиваемая длина истории

system_prompt = (
    "Ты — голосовой помощник на Windows 10. Запросы пользователя делятся на 2 типа: \n"
    "1. Простой вопрос \n"
    "2. Действие с компьютером \n"
    "Для первого типа дай краткий текстовый ответ. \n"
    "Для второго типа ответь в следующем формате: \n"
    "1. Укажи, является ли запрос простым (только true или false). True, если вопрос ясен без контекста (например: 'поставь музыку на паузу', 'открой проводник'). False для неоднозначных запросов (например: 'да, удали её', 'тут какая-то ошибка, исправь'). \n"
    "2. Сначала краткое текстовое описание действий — обязательно. \n"
    "3. Затем код на Python в ковычках ```. \n"
    "Не пиши никаких лишних символов (в том числе не нужно нумеровать пункты или использовать \"Описание:\" или \"Код:\") \n"
    "Всё взаимодействие с ПК происходит через Python. Пиши один завершённый рабочий скрипт, который будет автоматически выполняться. Не предлагай пользователю менять переменные. \n"
    "Ты можешь выполнять любые дествия с компьютером (открывать веб-сайты, менять настройки, создавать менять файлы и т.д.)"
)

# Загрузка стоп-слов
nltk.download('stopwords')  # Скачиваем список стоп-слов
stop_words = stopwords.words('russian')  # Получаем стоп-слова на русском
# Внедрение системного промта в историю
chat_history = [{"role": "system", "content": system_prompt}]

# Пути к файлам
db_path = Path("database.json")
scripts_dir = Path("saved_scripts")

# Загрузка JSON-файла с базой данных
def load_database():
    if db_path.exists():  # Используем метод exists() для проверки существования
        with open(db_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                return data if data else {}  # Если файл пуст, возвращаем пустой словарь
            except json.JSONDecodeError:  # Обработка случаев некорректного JSON
                return {}
    return {}

# Сохранение в базу данных
def save_database(database):
    db_path.parent.mkdir(parents=True, exist_ok=True)  # Создаём директорию, если её нет
    with open(db_path, "w", encoding="utf-8") as file:
        json.dump(database, file, ensure_ascii=False, indent=4)

# Функция хэширования текста
def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# Сохранение запроса и скрипта в базу
def save_request(request, description, code):
    database = load_database()
    request_hash = get_hash(request)
    
    # Сохраняем только если запрос отсутствует в базе
    if request_hash not in database:
        # Сохраняем код в отдельный файл
        scripts_dir.mkdir(parents=True, exist_ok=True)  # Создаём директорию для скриптов, если её нет
        script_path = scripts_dir / f"{request_hash}.py"  # Обновлённая переменная для пути к скрипту
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(code)
        
        # Обновляем базу данных
        database[request_hash] = {
            "request": request,
            "description": description,
            "script_path": str(script_path)
        }
        save_database(database)

# Оптимизированная функция для поиска похожих запросов
def find_similar_request(request):
    database = load_database()

    # Проверка наличия запросов в базе данных
    if not database:
        return None

    requests_list = [entry["request"] for entry in database.values()]
    requests_list.append(request)  # Добавляем текущий запрос для анализа

    # Создаем TF-IDF векторизатор
    vectorizer = TfidfVectorizer(stop_words=stop_words)  # Убираем стоп-слова
    tfidf_matrix = vectorizer.fit_transform(requests_list)

    # Получаем вектор для текущего запроса
    current_request_vector = tfidf_matrix[-1]

    # Сравниваем с сохранёнными запросами
    similarities = np.dot(tfidf_matrix[:-1], current_request_vector.T).toarray().flatten()
    best_match_index = np.argmax(similarities)

    # Проверяем, если схожесть выше порога
    if similarities[best_match_index] > 0.6:  # Порог схожести
        return database[list(database.keys())[best_match_index]]
    
    return None

def execute_saved_code(script_path, description):
    engine = pyttsx3.init()

    # Озвучиваем описание
    engine.say(description)
    engine.runAndWait()  # Убедитесь, что описание полностью озвучено перед выполнением кода

    # Проверяем существование и непустоту файла
    if os.path.exists(script_path) and os.path.getsize(script_path) > 0:
        # Выполняем код
        try:
            result = subprocess.run(
                ["python", script_path],
                check=True,
                text=True,
                capture_output=True,
                encoding=locale.getpreferredencoding()  # Использует системную кодировку
            )
            print("Результат выполнения кода:")
            print(result.stdout)

            # Проверяем, есть ли вывод перед озвучиванием
            if result.stdout.strip():  # Убираем пробелы
                engine.say(result.stdout)
                chat_history.append({"role": "python_code", "content": result.stdout})
                manage_chat_history()
            else:
                print("Код выполнен, но результат отсутствует.")
                engine.say("Код выполнен, но результат отсутствует.")
            
            engine.runAndWait()  # Убедитесь, что результат полностью озвучен

        except subprocess.CalledProcessError as e:
            print("Ошибка при выполнении кода:", e.stderr)
            chat_history.append({"role": "python_code_error", "content": e.stderr})
            manage_chat_history()
            engine.say("Произошла ошибка.")
            engine.runAndWait()
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

def recognize_and_process_audio(recognizer, frames):
    # Добавление искусственной тишины
    silence_duration = 0.5  # Длительность тишины в секундах
    silence_frame_count = int(silence_duration * RATE / CHUNK)  # Количество фреймов тишины
    silence_frame = b'\x00' * CHUNK  # Создание пустого фрейма

    # Добавляем нужное количество пустых фреймов
    for _ in range(silence_frame_count):
        frames.append(silence_frame)

    audio_data = sr.AudioData(b"".join(frames), RATE, 2)
    audio_length = len(frames) * CHUNK / RATE  # Длина записи в секундах

    if audio_length >= MIN_AUDIO_LENGTH:  # Проверка длины аудио
        print("Длина записи достаточна для распознавания.")
        try:
            text = recognizer.recognize_google(audio_data, language="ru-RU")
            print("Распознанный текст:", text)
            process_request(text)  # Обработка запроса
        except sr.UnknownValueError:
            print("Сервис не смог распознать текст.")
        except sr.RequestError:
            print("Ошибка подключения к сервису.")
    else:
        print("Запись слишком короткая, пропускаем.")


def process_request(text):
    text = text.lower().strip()  # Игнорирование регистра

    # Если похожий запрос не найден, обрабатываем новый
    if text.startswith("напечатай") or text.startswith("напиши"):
        processed_text = text.replace("напечатай", "").replace("напиши", "").strip()
        print("Обработка команды на печать...")
        print(f"Текст для печати: {processed_text}")
        keyboard.write(processed_text)

    elif text.startswith("приём"):
        processed_text = text.replace("приём", "").strip()

        # Проверка на существование похожего запроса в базе
        existing_entry = find_similar_request(processed_text)
        if existing_entry:
            print("Использование сохранённого ответа...")
            chat_history.append({"role": "user", "content": processed_text})
            manage_chat_history()

            # Выполнение сохранённого кода и добавление его в историю
            execute_saved_code(existing_entry["script_path"], existing_entry["description"])
            chat_history.append({"role": "assistant", "content": existing_entry["description"]})
            manage_chat_history()
            return  # Завершаем обработку здесь, чтобы не отправлять запрос в ChatGPT

        print("Отправка запроса в ChatGPT...")
        response = send_to_chatgpt(processed_text)
        print("Ответ ChatGPT:", response)

def manage_chat_history():
    """Управляет историей сообщений, удаляя старые сообщения при переполнении."""
    if len(chat_history) > MAX_HISTORY_LENGTH:
        chat_history.pop(1)  # Удаляем самое старое сообщение (игнорируя системное)

def send_to_chatgpt(user_message):
    client = Client()

    # Добавляем сообщение пользователя в историю
    chat_history.append({"role": "user", "content": user_message})
    manage_chat_history()

    # Отправка запроса к ChatGPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=chat_history,
    )

    assistant_response = response.choices[0].message.content.strip()

    # Проверка, нужно ли сохранять запрос
    save_flag = False
    response_lines = assistant_response.splitlines()

    # Находим первую непустую строку
    first_non_empty_line = next((line.strip().lower() for line in response_lines if line.strip()), None)
    print(first_non_empty_line)

    # Проверяем, содержится ли "true" или "false" в первой непустой строке
    if first_non_empty_line and ("true" in first_non_empty_line or "false" in first_non_empty_line):
        save_flag = "true" in first_non_empty_line
        # Удаляем первую непустую строку из assistant_response
        assistant_response = '\n'.join(line for line in response_lines if line.strip() != first_non_empty_line)

    print(save_flag)
    
    # Разделяем ответ на краткое описание и код
    response_parts = assistant_response.split('```', 1)
    short_description = response_parts[0].strip()

    print(short_description)

    # Извлекаем Python код между тройными кавычками, учитывая ```python
    python_code = ""
    if len(response_parts) > 1:
        python_code = response_parts[1].split('```', 1)[0].strip()
        if python_code.startswith("python"):
            python_code = python_code[len("python"):].strip()  # Удаление метки "python"

    # Сохраняем Python код во временный файл
    script_path = "output_code.py"
    with open(script_path, "w", encoding="utf-8") as code_file:
        code_file.write(python_code)

    # Выполняем и озвучиваем код
    execute_saved_code(script_path, short_description)

    # Сохраняем запрос и код в базу, если сохранение разрешено
    if save_flag:
        save_request(user_message, short_description, python_code)

    # Добавляем ответ ассистента в историю
    chat_history.append({"role": "assistant", "content": assistant_response})
    manage_chat_history()

    return assistant_response

def save_audio(frames):
    filename = "recorded_audio.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"Запись сохранена как {filename}")

def detect_and_record():
    recognizer = sr.Recognizer()
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Ожидание громкого звука...")

    frames = []
    is_recording = False
    silence_start = None

    while True:
        data = stream.read(CHUNK)
        rms = audioop.rms(data, 2)  # Рассчитываем уровень громкости

        # Начинаем запись, если громкость выше стартового порога
        if rms > THRESHOLD_START and not is_recording:
            print("Громкий звук обнаружен, начинается запись...")
            is_recording = True
            frames.append(data)
            silence_start = None

        # Продолжаем запись при достижении более низкого порога
        elif is_recording:
            if rms > THRESHOLD_CONTINUE:
                frames.append(data)
            if rms <= THRESHOLD_START:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("Тишина, завершаем запись.")
                    break
            else:
                silence_start = None

    # Обрабатываем записи только если есть данные
    if frames:
        print("Обработка записи...")
        # save_audio(frames)  # Сохранение аудиозаписи
        recognize_and_process_audio(recognizer, frames)  # Обработка записанных фреймов

    stream.stop_stream()
    stream.close()
    p.terminate()

# Запуск функции
while True:
    logging.basicConfig(
    filename="app_errors.log",  # Имя файла для сохранения логов
    level=logging.ERROR,  # Устанавливаем уровень для ошибок
    format="%(asctime)s - %(levelname)s - %(message)s",  # Формат логов с указанием времени
    )

    try:
        detect_and_record()
    except Exception as e:
        logging.error("Общая ошибка: %s", e)  # Логирование исключения с описанием
        print("Произошла ошибка, подробности записаны в app_errors.log")
