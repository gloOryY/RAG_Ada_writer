from __future__ import annotations  # включаем отложенные аннотации типов

from dataclasses import dataclass  # для декларативного описания источника данных
from pathlib import Path  # объектно-ориентированная работа с путями
from typing import Iterable, List  # типы для читаемых аннотаций

from langchain_community.document_loaders import PyPDFLoader  # загрузчик PDF в объекты Document
from langchain_core.documents import Document  # стандартная структура документа в LangChain


@dataclass(frozen=True)  # неизменяемое описание источника знаний
class KBSource:  # параметры загрузки PDF
    """
    Описание источника знаний (папка с PDF).
    """
    pdf_dir: Path  # директория, где лежат PDF
    pdf_mode: str  # "page" или "single" — режим разбиения документа


def iter_pdf_paths(pdf_dir: Path) -> Iterable[Path]:  # итерируем все PDF-файлы рекурсивно
    for p in pdf_dir.rglob("*.pdf"):  # ищем файлы с расширением .pdf в подпапках
        if p.is_file():  # проверяем, что найденный путь — именно файл
            yield p  # отдаём путь вызывающему коду


def load_pdfs(source: KBSource) -> List[Document]:  # загружает PDF в список Document
    """
    Загружает все PDF как список langchain Document.

    PyPDFLoader обычно режет PDF на документы постранично (mode="page").
    Если mode="single", грузит как один поток текста (удобно, когда абзацы
    не хочется рвать на границах страниц).
    """
    if source.pdf_mode not in {"page", "single"}:  # валидация допустимых режимов
        raise ValueError("PDF_MODE должен быть 'page' или 'single'")  # сообщаем о некорректном значении

    docs: List[Document] = []  # аккумулируем загруженные документы
    for pdf_path in iter_pdf_paths(source.pdf_dir):  # перебираем все файлы PDF
        loader = PyPDFLoader(str(pdf_path), mode=source.pdf_mode)  # создаём загрузчик для файла
        docs.extend(loader.load())  # добавляем результаты загрузки (один или несколько документов)
    return docs  # возвращаем полный список документов
