from __future__ import annotations  # включаем отложенные аннотации типов

import argparse  # парсер аргументов командной строки для режима чата

from langchain_huggingface import HuggingFaceEmbeddings  # локальные эмбеддинги (SentenceTransformers)
from langchain_groq import ChatGroq  # клиент Groq LLM

from .config import load_settings  # загрузка настроек (.env)
from .rag import RAGConfig, answer_with_rag  # конфиг RAG и функция ответа
from .vectorstore import VectorStoreSpec, open_chroma  # доступ к Chroma и открытие коллекции


def main() -> None:  # интерактивный чат
    parser = argparse.ArgumentParser(description="Interactive Ada RAG chat (Groq + Chroma)")  # описываем CLI
    parser.add_argument("--k", type=int, default=None, help="Сколько чанков подтягивать (top_k)")  # переопределение top_k
    args = parser.parse_args()  # парсим аргументы

    s = load_settings()  # читаем настройки из окружения

    # Embeddings должны быть теми же, что и при индексации
    embeddings = HuggingFaceEmbeddings(model_name=s.embedding_model)  # эмбеддинги должны совпадать с этапом индексации

    vs = open_chroma(  # открываем/создаём коллекцию Chroma
        VectorStoreSpec(persist_directory=s.chroma_dir, collection_name=s.chroma_collection),  # спецификация
        embeddings,  # функция преобразования текста в вектор
    )
    retriever = vs.as_retriever(search_kwargs={"k": args.k or s.top_k})  # создаём ретривер с нужным количеством результатов

    # LLM Groq (ключ берётся из env GROQ_API_KEY автоматически, но мы храним его в настройках)
    llm = ChatGroq(  # инициализируем LLM-клиент Groq
        model=s.groq_model,  # выбранная модель
        temperature=s.groq_temperature,  # степень случайности
        max_tokens=s.groq_max_tokens,  # ограничение длины ответа
        api_key=s.groq_api_key,  # явно прокидываем ключ (удобно для отладки)
    )

    cfg = RAGConfig(top_k=args.k or s.top_k)  # конфиг для ограничения количества документов

    print("Ada RAG chat. Напиши вопрос. Для выхода: exit / quit")
    while True:  # основной цикл ввода/вывода
        q = input("\n> ").strip()  # читаем вопрос пользователя
        if not q:  # пропускаем пустые строки
            continue
        if q.lower() in {"exit", "quit"}:  # команда выхода
            break

        try:  # безопасно обрабатываем ошибки
            a = answer_with_rag(llm, retriever, cfg, q)  # получаем ответ от RAG-пайплайна
            print("\n" + a)  # печатаем ответ
        except Exception as e:  # ловим исключения, чтобы не падать
            print(f"\n[ERROR] {e}")  # печатаем ошибку


if __name__ == "__main__":  # точка входа при запуске как скрипта
    main()  # запускаем чат 
