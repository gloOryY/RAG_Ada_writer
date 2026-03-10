from __future__ import annotations  # включаем отложенные аннотации типов

import argparse  # парсим аргументы командной строки
import shutil  # операции с файлами/папками (удаление директории базы)
from pathlib import Path  # работа с путями (удобно для кроссплатформенности)

from langchain_huggingface import HuggingFaceEmbeddings  # новый пакет (вместо langchain_community) [web:138]
from langchain_text_splitters import RecursiveCharacterTextSplitter  # отдельный пакет splitters [web:32]

from .config import load_settings  # читаем настройки (.env)
from .kb_loaders import KBSource, load_pdfs  # загрузка PDF в документы
from .vectorstore import VectorStoreSpec, open_chroma  # доступ к Chroma векторному хранилищу

# Chroma/SQLite часто имеет лимит max_batch_size (например 5461), поэтому пишем чанки батчами. [web:76]
SAFE_BATCH_SIZE = 4000  # гарантированно меньше 5461, безопасное значение по умолчанию


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:  # создаёт рекурсивный сплиттер текста
    """
    Делит текст рекурсивно:
    - сперва пытается резать по "крупным" разделителям (абзацы/строки),
    - если кусок всё ещё слишком большой, режет дальше (по пробелам и т.п.).
    chunk_overlap нужен, чтобы на границах чанков не терялись важные определения.
    """
    return RecursiveCharacterTextSplitter(  # конфигурация правил разбиения текста на чанки
        chunk_size=chunk_size,  # максимально допустимый размер чанка
        chunk_overlap=chunk_overlap,  # перекрытие между соседними чанками
        separators=["\n\n", "\n", " ", ""],  # прогрессивные разделители от крупных к мелким
        is_separator_regex=False,  # разделители — обычные строки, не регулярные выражения
        length_function=len,  # функция измерения длины (по символам)
    )


def batched(iterable, batch_size: int):  # генератор, который делит список на батчи указанного размера
    """Простой генератор батчей: [0:batch_size], [batch_size:2*batch_size], ..."""
    n = len(iterable)  # общее количество элементов
    for start in range(0, n, batch_size):  # шагаем по списку с шагом batch_size
        yield start, iterable[start : start + batch_size]  # отдаём индекс начала и сам батч


def main() -> None:  # главная функция индексации PDF в Chroma
    parser = argparse.ArgumentParser(description="Index Ada textbooks (PDF) into Chroma")  # описываем CLI
    parser.add_argument("--pdf-dir", default="./data/pdfs", help="Папка с PDF учебниками")  # где искать PDF
    parser.add_argument("--reset", action="store_true", help="Удалить старую базу Chroma перед индексацией")  # пересоздать БД
    parser.add_argument("--batch-size", type=int, default=SAFE_BATCH_SIZE, help="Размер батча для upsert в Chroma")  # размер батча
    args = parser.parse_args()  # разбираем аргументы

    s = load_settings()  # читаем настройки из .env/окружения

    pdf_dir = Path(args.pdf_dir)  # путь к директории с PDF
    pdf_dir.mkdir(parents=True, exist_ok=True)  # создаём директорию, если её нет

    # Если хочешь пересоздать БД "с нуля" — удаляем папку Chroma.
    if args.reset:  # по флагу --reset очищаем базу
        shutil.rmtree(s.chroma_dir, ignore_errors=True)  # удаляем директорию, игнорируя ошибки

    # 1) Загружаем PDF -> список Document
    src = KBSource(pdf_dir=pdf_dir, pdf_mode=s.pdf_mode)  # описываем источник знаний
    raw_docs = load_pdfs(src)  # загружаем PDF -> список Document
    if not raw_docs:  # если ничего не нашли — сообщаем
        raise RuntimeError(
            f"В папке {pdf_dir} не найдено ни одного PDF. Положи учебники в data/pdfs и повтори."
        )

    # 2) Режем на чанки
    splitter = build_splitter(s.chunk_size, s.chunk_overlap)  # создаём сплиттер под конфиг
    chunks = splitter.split_documents(raw_docs)  # делим документы на чанки

    # 3) Создаём embeddings (локальная модель SentenceTransformers)
    # Важно: используй тот же embedding_model при chat, иначе поиск будет “кривой”. [web:138]
    embeddings = HuggingFaceEmbeddings(model_name=s.embedding_model)  # локальные эмбеддинги (SentenceTransformers)

    # 4) Открываем/создаём Chroma (persist_directory = папка на диске)
    vs = open_chroma(  # создаём/открываем коллекцию Chroma
        VectorStoreSpec(persist_directory=s.chroma_dir, collection_name=s.chroma_collection),  # спецификация БД
        embeddings,  # функция эмбеддингов
    )

    # 5) Записываем чанки батчами, чтобы не упереться в max_batch_size (например 5461) [web:76]
    batch_size = int(args.batch_size)  # приводим размер батча к int
    if batch_size <= 0:  # минимальная проверка на валидность
        raise ValueError("--batch-size должен быть > 0")  # сообщаем о неверном значении
    if batch_size > 5400:  # предупреждаем о риске превышения лимита SQLite
        # Не запрещаем, но предупреждаем: у тебя уже вылетало на 6323 > 5461.
        print(f"[WARN] batch_size={batch_size} может быть слишком большим для Chroma/SQLite лимита. [web:76]")  # лог предупреждения

    total = len(chunks)  # общее количество чанков
    for start, batch in batched(chunks, batch_size):  # перебираем батчи
        vs.add_documents(batch)  # записываем батч в Chroma
        done = min(start + batch_size, total)  # считаем прогресс
        print(f"[OK] Upserted {done}/{total} chunks")  # выводим статус индексации

    print(f"[OK] PDF docs loaded: {len(raw_docs)}")  # количество загруженных PDF-документов
    print(f"[OK] Chunks created:   {len(chunks)}")  # количество созданных чанков
    print(f"[OK] Chroma dir:      {s.chroma_dir}")  # путь к директории базы Chroma
    print(f"[OK] Collection:      {s.chroma_collection}")  # имя коллекции


if __name__ == "__main__":  # точка входа при запуске как скрипта
    main()  # стартуем процесс индексации
