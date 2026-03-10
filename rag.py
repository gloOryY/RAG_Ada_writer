from __future__ import annotations  # включаем отложенные аннотации типов

from dataclasses import dataclass  # для компактного описания конфигурации
from typing import List  # используем List для совместимости с типами документов

from langchain_core.documents import Document  # тип объекта документа (контент + метаданные)
from langchain_groq import ChatGroq  # LLM-клиент от Groq для генерации ответа


@dataclass(frozen=True)  # конфигурация RAG-ответа неизменяема
class RAGConfig:  # параметры процесса ответа
    top_k: int  # сколько документов брать из векторного поиска


def format_context(docs: List[Document]) -> str:  # превращает список документов в читаемую строку контекста
    """
    Форматируем контекст так, чтобы:
    - модель видела источники (какой PDF, какая страница);
    - контекст был “читабельным”.
    """
    parts: List[str] = []  # накапливаем отформатированные куски
    for i, d in enumerate(docs, start=1):  # пронумеровываем найденные документы
        src = d.metadata.get("source", "unknown_source")  # путь к PDF-источнику
        page = d.metadata.get("page", None)  # номер страницы (если есть)
        page_str = f", page={page}" if page is not None else ""  # добавляем страницу в подпись
        parts.append(f"[{i}] source={src}{page_str}\n{d.page_content}")  # текст документа после подписи
    return "\n\n---\n\n".join(parts)  # разделяем источники визуально


def build_messages(context: str, user_query: str) -> list[tuple[str, str]]:  # собирает сообщения для LLM (system + human)
    """
    Делаем строгую инструкцию, чтобы:
    - модель писала Ada-код;
    - ссылалась на контекст;
    - если контекста недостаточно — честно говорила, что добавить.
    """
    system = (  # строгая инструкция для модели
        "Ты помощник по языку Ada.\n"
        "Твоя цель: написать корректный Ada-код по запросу пользователя.\n"
        "Правила:\n"
        "1) Используй КОНТЕКСТ как основной источник фактов/синтаксиса.\n"
        "2) Если в контексте нет нужной информации, прямо напиши, чего не хватает, и задай 1 уточняющий вопрос.\n"
        "3) Если пишешь код: возвращай полный компилируемый пример (procedure Main) и минимум пояснений."
    )

    human = (  # сообщение пользователя + сформированный контекст
        f"КОНТЕКСТ:\n{context}\n\n"
        f"ЗАПРОС:\n{user_query}\n"
    )

    return [("system", system), ("human", human)]  # последовательность сообщений для ChatGroq


def answer_with_rag(llm: ChatGroq, retriever, cfg: RAGConfig, user_query: str) -> str:  # полный RAG-процесс ответа
    """
    Основная функция ответа:
      user_query -> retrieve top_k -> prompt -> llm -> answer
    """
    docs = retriever.invoke(user_query)  # ищем релевантные куски текста по запросу
    docs = docs[: cfg.top_k]  # ограничиваем количество документов

    context = format_context(docs)  # формируем читаемый контекст
    messages = build_messages(context, user_query)  # собираем промпт для модели
    return llm.invoke(messages).content  # генерируем ответ и возвращаем текст
