from __future__ import annotations  # включает «отложенные» аннотации типов для совместимости

import os  # стандартный модуль для работы с переменными окружения
from dataclasses import dataclass  # удобный способ описать структуру данных (настройки)
from dotenv import load_dotenv  # функция для загрузки переменных из файла .env

from pathlib import Path  # кроссплатформенная работа с путями

 


@dataclass(frozen=True)  # делаем класс неизменяемым после создания
class Settings:  # контейнер всех параметров проекта
    # Groq
    groq_api_key: str  # ключ доступа к Groq API
    groq_model: str  # выбранная LLM-модель Groq
    groq_temperature: float  # степень случайности генерации
    groq_max_tokens: int  # максимум токенов в ответе

    # Storage
    chroma_dir: str  # директория на диске для хранения Chroma DB
    chroma_collection: str  # имя коллекции (таблицы) в Chroma

    # Embeddings
    embedding_model: str  # модель эмбеддингов SentenceTransformers

    # Chunking / Retrieval
    chunk_size: int  # размер текстового чанка при индексации
    chunk_overlap: int  # перекрытие между соседними чанками
    top_k: int  # сколько релевантных чанков подтягивать на запрос

    # PDF loader
    pdf_mode: str  # режим загрузки PDF: "page" (по страницам) или "single" (единый поток)


def load_settings() -> Settings:  # загружает настройки из окружения и валидирует ключ Groq
    """
    Читает настройки из .env / переменных окружения.

    Важно:
    - GROQ_API_KEY должен быть установлен, иначе LLM не сможет работать.
    """
    # Сначала загружаем .env из пакета (ada_rag/.env) как дефолты,
    # затем .env из корня проекта, позволяя ему переопределять значения.
    pkg_env = Path(__file__).resolve().parent / ".env"
    root_env = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=pkg_env, override=False)
    load_dotenv(dotenv_path=root_env, override=True)

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()  # читаем ключ Groq и убираем пробелы
    if not groq_api_key:  # если ключ пустой — считаем это ошибкой конфигурации
        raise RuntimeError(
            "Не найден GROQ_API_KEY. Создай .env (см. .env.example) или задай переменную окружения."
        )

    return Settings(  # формируем объект настроек с дефолтами
        groq_api_key=groq_api_key,  # обязательный ключ Groq
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),  # модель Groq по умолчанию
        groq_temperature=float(os.getenv("GROQ_TEMPERATURE", "0")),  # детерминированная генерация по умолчанию
        groq_max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "800")),  # ограничение длины ответа
        chroma_dir=os.getenv("CHROMA_DIR", "./chroma_db"),  # папка для базы Chroma
        chroma_collection=os.getenv("CHROMA_COLLECTION", "ada_textbooks"),  # имя коллекции для учебников
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),  # локальная модель эмбеддингов
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),  # стандартный размер чанка
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),  # перекрытие для сохранения контекста на границах
        top_k=int(os.getenv("TOP_K", "5")),  # сколько чанков отдаёт retriever
        pdf_mode=os.getenv("PDF_MODE", "page").strip().lower(),  # режим загрузки PDF с нормализацией
    )
