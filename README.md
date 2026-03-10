# Ada RAG (LangChain + Groq + Chroma)

---

## Архитектура (коротко)

1) Ingestion (индексация):
- `PyPDFLoader` читает PDF -> список `Document` (обычно по страницам).
- `RecursiveCharacterTextSplitter` режет текст на чанки (кусочки).
- Локальные embeddings (SentenceTransformers) превращают чанки в векторы.
- Chroma сохраняет векторы + текст на диск в `CHROMA_DIR`.

2) Answer (ответ на запрос):
- По запросу пользователя делаем semantic search в Chroma (top_k чанков).
- Склеиваем найденные чанки в “контекст”.
- Отправляем контекст + запрос в Groq LLM через LangChain `ChatGroq`.
- Модель генерирует Ada-код.

---

## Требования

- Python 3.10+ 
- Аккаунт Groq + API key

---

## Установка

### 1) Склонировать/создать папку проекта
Структура должна быть такой:

ada-rag/
  ada_rag/
  data/pdfs/
  requirements.txt
  .env.example
  README.md

### 2) Виртуальное окружение

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt

### 3) Запуск
Обычный запуск:
```powershell
python -m ada_rag.ingest

Если хочешь пересоздать базу (удалить старую Chroma):
```powershell
python -m ada_rag.ingest --reset

Запуск чата
```powershell
python -m ada_rag.chat

### 4) Описание файлов проекта 

1. config.py — центр управления настройками
Это единственное место, где хранятся все параметры проекта. Использует две концепции
@dataclass(frozen=True) — класс Settings
frozen=True делает объект неизменяемым после создания — защита от случайной модификации настроек в рантайме. Поля разбиты на группы:
Groq: groq_api_key, groq_model (по умолчанию llama-3.3-70b-versatile), groq_temperature=0 (детерминированный ответ), groq_max_tokens=800
Storage: chroma_dir=./chroma_db, chroma_collection=ada_textbooks
Embeddings: embedding_model=sentence-transformers/all-MiniLM-L6-v2
Chunking: chunk_size=1200, chunk_overlap=150, top_k=5
PDF: pdf_mode=page
load_settings() — функция загрузки
Грузит значения из двух .env файлов — сначала ada_rag/.env (дефолты пакета), затем корневой .env (переопределяет). Если GROQ_API_KEY не задан — сразу бросает RuntimeError, чтобы не запускать систему с пустым ключом.
​

2. vectorstore.py — обёртка над Chroma
Абстрагирует работу с векторной базой данных.
VectorStoreSpec — dataclass с двумя полями:
persist_directory — путь к папке на диске, куда Chroma сохраняет данные
collection_name — имя «таблицы» внутри базы (аналог namespace)
open_chroma(spec, embeddings) — создаёт или открывает коллекцию Chroma. При первом запуске база создаётся, при повторном — просто открывается. Ключевой момент: та же функция embeddings должна использоваться и при записи, и при поиске — иначе векторы будут несовместимы.
​

3. kb_loaders.py — загрузчик PDF
Отвечает за превращение PDF-файлов в объекты Document.
Что такое Document в LangChain?
Это объект с двумя полями:
page_content — сам текст
metadata — словарь с метаданными (source, page, и т.д.)
KBSource — dataclass с путём к папке и режимом загрузки PDF.
iter_pdf_paths(pdf_dir) — рекурсивно (через .rglob("*.pdf")) ищет все PDF в папке и подпапках.
load_pdfs(source) — для каждого PDF создаёт PyPDFLoader с двумя режимами:
mode="page" — каждая страница PDF становится отдельным Document (стандартно)
mode="single" — весь PDF — один Document (полезно, когда абзацы нельзя рвать по границам страниц)

4. ingest.py — пайплайн индексации (ЭТАП 1)
Это скрипт-оркестратор первого этапа. Запускается один раз (или при обновлении данных). Содержит 5 шагов:
build_splitter(chunk_size, chunk_overlap)
Создаёт RecursiveCharacterTextSplitter — умный сплиттер, который режет текст иерархически:
Сначала по двойному переносу строки (\n\n — абзацы)
Затем по одному переносу (\n — строки)
Затем по пробелам ( )
В крайнем случае — по символам ("")
chunk_overlap=150 означает, что 150 символов с конца предыдущего чанка войдут в начало следующего — это помогает не потерять контекст определений на границах.
batched(iterable, batch_size)
Генератор батчей. Chroma/SQLite имеет лимит max_batch_size ≈ 5461, поэтому безопасное значение SAFE_BATCH_SIZE = 4000.

main() — 5 шагов:
1. Загрузка аргументов CLI (--pdf-dir, --reset, --batch-size)
2. Опционально удаляем старую Chroma (shutil.rmtree)
3. load_pdfs() → список Document
4. splitter.split_documents() → список чанков
5. HuggingFaceEmbeddings + open_chroma → vs.add_documents() батчами

5. rag.py — мозг системы ответов (ЭТАП 2)
Содержит три функции, которые вместе составляют RAG-пайплайн:
format_context(docs)
Превращает список Document в форматированную строку для промпта. Каждый документ получает заголовок с источником и номером страницы:

'''
source=data/pdfs/ada_textbook.pdf, page=42
<текст чанка>
---
source=...
''' 

Это позволяет модели «видеть», откуда взята информация.
build_messages(context, user_query)
Составляет промпт из двух сообщений:
system — строгая инструкция: «ты помощник по Ada, пиши компилируемый код, используй контекст, если не хватает — скажи чего не хватает»
human — блок КОНТЕКСТ + блок ЗАПРОС
answer_with_rag(llm, retriever, cfg, user_query)

Главная функция всего RAG-процесса:

user_query → retriever.invoke() → docs[:top_k] → format_context() → build_messages() → llm.invoke() → .content
retriever.invoke(user_query) делает семантический поиск: вопрос превращается в вектор той же моделью эмбеддингов, и Chroma находит top_k ближайших по косинусному расстоянию чанков.
​

6. chat.py — интерфейс пользователя
Склеивает все модули в интерактивный чат в терминале:
Читает настройки через load_settings()
Загружает ту же модель эмбеддингов (важно: должна совпадать с ingest.py!)
Открывает Chroma через open_chroma()
Создаёт retriever = vs.as_retriever(search_kwargs={"k": top_k})
Инициализирует ChatGroq с ключом и параметрами
Запускает while True цикл: читает вопрос → вызывает answer_with_rag() → печатает ответ
Выход — команды exit или quit. Ошибки перехватываются через try/except, чтобы чат не падал при сбоях сети или API.

Общая схема потока данных

ЭТАП 1 (ingest.py) — один раз:
PDF-файлы
  └─► kb_loaders.py (PyPDFLoader) → [Document, ...]
        └─► ingest.py (RecursiveCharacterTextSplitter) → [чанк, ...]
              └─► HuggingFaceEmbeddings → [вектор, ...]
                    └─► vectorstore.py (Chroma) → сохранено на диск

ЭТАП 2 (chat.py) — каждый вопрос:
Вопрос пользователя
  └─► HuggingFaceEmbeddings → вектор вопроса
        └─► Chroma (семантический поиск) → top_k чанков
              └─► rag.py (format_context + build_messages) → промпт
                    └─► ChatGroq (Groq API) → Ada-код ✓

