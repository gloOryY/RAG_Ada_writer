from __future__ import annotations  # включаем отложенные аннотации типов

from dataclasses import dataclass  # декларативное описание конфигурации хранилища

from langchain_chroma import Chroma  # адаптер LangChain для работы с Chroma
from langchain_core.embeddings import Embeddings  # протокол/интерфейс эмбеддингов

@dataclass(frozen=True)  # неизменяемая спецификация для подключения к БД
class VectorStoreSpec:  # параметры хранилища
    """
    Настройки подключения к Chroma.
    """
    persist_directory: str  # путь на диске, где Chroma хранит данные
    collection_name: str  # имя коллекции внутри Chroma


def open_chroma(spec: VectorStoreSpec, embeddings: Embeddings) -> Chroma:  # создаёт/открывает коллекцию Chroma
    """
    Открывает (или создаёт при первом запуске) коллекцию Chroma в локальной директории.
    """
    return Chroma(  # инициализация клиента Chroma с заданными параметрами
        collection_name=spec.collection_name,  # имя коллекции
        embedding_function=embeddings,  # функция, которая преобразует текст в вектор
        persist_directory=spec.persist_directory,  # директория для сохранения данных
    )
