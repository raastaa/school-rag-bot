# bot.py
import os
import sys
import asyncio
import shutil
import subprocess
import tempfile
import html
from typing import List

from dotenv import load_dotenv
load_dotenv()

# --- sys.path safety ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- aiogram ---
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    FSInputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
)
from aiogram.enums import ChatAction
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

# --- our modules ---
from retrieval_local import (
    retrieve_local_hits,
    format_answer_from_payload,
    preview_from_payload,
)
from ingest.pdf_ingest import ingest_pdf           # админ: одиночный PDF
from ingest.ingest_generic import ingest_path      # индексация папки (teach/)
from store_qdrant import get_client                # /clear_index
from db_local import (
    init_db, upsert_user, insert_question, mark_answered,
    log_unanswered, log_answer_score
)

# Этапы 2/3 (поиск по сайту / по вебу) — должны быть в проекте
from retrieve_site_live import retrieve_site_live
from retrieve_web_live  import retrieve_web_live

# -------------------- настройки --------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не задан в .env")

ADMIN_IDS = {int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()}
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "school_docs")
TEACH_DIR = os.getenv("TEACH_DIR", "./teach")

bot = Bot(BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

# Храним для пользователей список найденных локальных ответов
pending_local: dict[int, list] = {}

# -------------------- вспомогалки UI/админ --------------------
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

def admin_kb():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="➕ Добавить документ")],
        ],
        resize_keyboard=True
    )

def _split_long(text: str, limit: int = 4000) -> list[str]:
    """Режем длинное сообщение для Telegram (лимит ~4096)."""
    if not text:
        return [""]
    out, cur, cur_len = [], [], 0
    for para in text.split("\n\n"):
        add = para + "\n\n"
        if cur_len + len(add) > limit and cur:
            out.append("".join(cur).rstrip())
            cur, cur_len = [add], len(add)
        else:
            cur.append(add); cur_len += len(add)
    if cur:
        out.append("".join(cur).rstrip())
    return out

# -------------------- отправка источников (+ DOCX→PDF) --------------------
CONVERT_DIR = os.path.join("outputs", "converted")
os.makedirs(CONVERT_DIR, exist_ok=True)

def docx_to_pdf(src_path: str) -> str | None:
    """Конвертирует .docx -> .pdf через LibreOffice (headless)."""
    if not os.path.isfile(src_path):
        return None
    soffice = shutil.which("soffice")
    if not soffice:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="docx2pdf_") as tmpdir:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf", "--outdir", tmpdir, src_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            base = os.path.splitext(os.path.basename(src_path))[0] + ".pdf"
            tmp_pdf = os.path.join(tmpdir, base)
            if not os.path.exists(tmp_pdf):
                return None
            dst_pdf = os.path.join(CONVERT_DIR, base)
            shutil.move(tmp_pdf, dst_pdf)
            return dst_pdf
    except Exception:
        return None

async def send_files(m: Message, paths: List[str], limit: int = 3):
    """Отправляем файлы-источники (до limit штук). DOCX конвертируем в PDF."""
    count = 0
    for p in paths:
        if count >= limit:
            await m.answer(f"Показано первых {limit} файлов-источников.", parse_mode="HTML")
            break
        try:
            ext = os.path.splitext(p)[1].lower()
            to_send_path = p
            caption = os.path.basename(p)

            if ext == ".docx":
                pdf_path = docx_to_pdf(p)
                if pdf_path and os.path.exists(pdf_path):
                    to_send_path = pdf_path
                    caption = os.path.basename(pdf_path)
                else:
                    await m.answer("Не удалось преобразовать DOCX в PDF, отправляю исходный файл.", parse_mode="HTML")

            await bot.send_chat_action(m.chat.id, ChatAction.UPLOAD_DOCUMENT)
            await m.answer_document(FSInputFile(to_send_path), caption=caption)
            count += 1
        except Exception:
            await m.answer(f"Не удалось отправить файл: {p}", parse_mode="HTML")

# -------------------- FSM: добавление документа (PDF) --------------------
class AddDoc(StatesGroup):
    waiting_title = State()
    waiting_file = State()

@router.message(Command('cancel'), AddDoc)
async def cancel_add_doc(m: Message, state: FSMContext):
    await state.clear()
    await m.answer("Добавление документа отменено.", parse_mode="HTML")

@router.message(F.text == "➕ Добавить документ")
async def add_doc(m: Message, state: FSMContext):
    if not is_admin(m.from_user.id):
        return await m.answer("Эта функция доступна только администратору.", parse_mode="HTML")
    await state.set_state(AddDoc.waiting_title)
    await m.answer(
        "Введите название документа (как показывать пользователям). Командой /cancel можно прервать процесс.",
        parse_mode="HTML",
    )

@router.message(AddDoc.waiting_title, F.text)
async def got_title(m: Message, state: FSMContext):
    await state.update_data(title=m.text.strip())
    await state.set_state(AddDoc.waiting_file)
    await m.answer(
        "Пришлите PDF-файл документом (не фото). Командой /cancel можно прервать процесс.",
        parse_mode="HTML",
    )

@router.message(AddDoc.waiting_file, F.document)
async def got_file(m: Message, state: FSMContext):
    if not is_admin(m.from_user.id):
        return await m.answer("Эта функция доступна только администратору.", parse_mode="HTML")
    doc = m.document
    if not doc.mime_type or "pdf" not in doc.mime_type.lower():
        return await m.answer("Нужен именно PDF-файл. Пришлите ещё раз как документ.", parse_mode="HTML")

    data = await state.get_data()
    title = data.get("title") or doc.file_name or "Без названия"

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", doc.file_name or f"upload_{doc.file_unique_id}.pdf")
    await m.answer("Файл получен, начинаю индексацию… (это может занять несколько минут)", parse_mode="HTML")
    try:
        await bot.download(doc, destination=file_path)
        res = await ingest_pdf(file_path, title=title, source_label=doc.file_name)
        await m.answer(
            "Готово. Документ добавлен в локальную базу.\n"
            f"Название: {res.get('document')}\n"
            f"Группа: {res.get('source_group')}\n"
            f"Страниц: {res.get('pages')}\n"
            f"Чанков: {res.get('chunks')}\n"
            f"Файл: {res.get('file')}",
            parse_mode="HTML"
        )
    except Exception as e:
        await m.answer(f"Ошибка при индексации: {e}", parse_mode="HTML")
    finally:
        await state.clear()

@router.message(AddDoc.waiting_file)
async def not_file(m: Message):
    await m.answer("Пришлите, пожалуйста, PDF-файл документом.", parse_mode="HTML")

# -------------------- команды админа --------------------
@router.message(CommandStart())
async def cmd_start(m: Message):
    if is_admin(m.from_user.id):
        await m.answer(
            "Здравствуйте! Отправьте вопрос — я пришлю найденные фрагменты текстом и приложу исходные файлы.\n"
            "Команды:\n"
            "• /help — показать список команд\n"
            "• /clear_index — очистить локальный индекс Qdrant\n"
            "• /ingest_teach — проиндексировать все файлы из папки teach/ (source_group=teach)",
            reply_markup=admin_kb(), parse_mode="HTML"
        )
    else:
        await m.answer(
            "Здравствуйте! Отправьте свой вопрос — я пришлю найденные фрагменты из локальной базы и приложу файлы.\n"
            "Напишите /help для списка команд.",
            parse_mode="HTML"
        )

@router.message(Command('help'))
async def cmd_help(m: Message):
    is_adm = is_admin(m.from_user.id)
    text = (
        "Доступные команды:\n"
        "• /start — приветственное сообщение\n"
        "• /help — показать эту справку"
    )
    if is_adm:
        text += (
            "\n• /clear_index — очистить локальный индекс Qdrant"
            "\n• /ingest_teach — проиндексировать все файлы из папки teach/ (source_group=teach)"
        )
        await m.answer(text, reply_markup=admin_kb(), parse_mode="HTML")
    else:
        await m.answer(text, parse_mode="HTML")

@router.message(Command("clear_index"))
async def cmd_clear_index(m: Message):
    if not is_admin(m.from_user.id):
        return await m.answer("Эта команда доступна только администратору.", parse_mode="HTML")
    try:
        cli = get_client()
        cols = [c.name for c in cli.get_collections().collections]
        if QDRANT_COLLECTION in cols:
            cli.delete_collection(QDRANT_COLLECTION)
        await m.answer("Локальный индекс очищен. Можно загружать документы заново.", parse_mode="HTML")
    except Exception as e:
        await m.answer(f"Не удалось очистить индекс: {e}", parse_mode="HTML")

@router.message(Command("ingest_teach"))
async def cmd_ingest_teach(m: Message):
    if not is_admin(m.from_user.id):
        return await m.answer("Эта команда доступна только администратору.", parse_mode="HTML")
    if not os.path.isdir(TEACH_DIR):
        return await m.answer(f"Папка '{TEACH_DIR}' не найдена. Создайте её и положите туда файлы.", parse_mode="HTML")
    await m.answer(f"Начинаю индексацию файлов из {TEACH_DIR} (source_group=teach)…", parse_mode="HTML")
    try:
        res = await ingest_path(TEACH_DIR, source_group="teach")
        await m.answer(f"Готово. Файлов: {res.get('files')}, чанков: {res.get('chunks')}", parse_mode="HTML")
    except Exception as e:
        await m.answer(f"Ошибка при индексации teach/: {e}", parse_mode="HTML")

# -------------------- обработка вопроса пользователя --------------------
@router.message(F.text & ~F.text.in_({"➕ Добавить документ"}))
async def handle_question(m: Message):
    q = m.text.strip()

    # Сброс предыдущих локальных выдач для пользователя
    pending_local.pop(m.from_user.id, None)

    # 0) БД: логируем пользователя и вопрос
    from db_local import DB_PATH  # только ради удобной отладки
    tg = m.from_user
    user_id = await asyncio.to_thread(upsert_user, tg.id, tg.username, tg.first_name, tg.last_name)
    question_id = await asyncio.to_thread(insert_question, user_id, q)

    # сообщение о ходе поиска
    msg = await m.answer("Ищу…", parse_mode="HTML")

    # Этап 1 — локальная база
    await m.answer("Ищу по локальной базе…", parse_mode="HTML")
    await bot.send_chat_action(m.chat.id, ChatAction.TYPING)
    hits, diag = await retrieve_local_hits(q, top_k=5, prefer_spravochnik=False)

    # Логируем все оценки (принятые и отфильтрованные)
    for payload, score in (diag.get("passed") or []):
        await asyncio.to_thread(log_answer_score, question_id, payload, score, True)
    for payload, score in (diag.get("rejected") or []):
        await asyncio.to_thread(log_answer_score, question_id, payload, score, False)

    if hits:
        await msg.edit_text("Нашёл ответы в локальной базе", parse_mode="HTML")
        await asyncio.to_thread(mark_answered, question_id, "local")
        pending_local[m.from_user.id] = hits
        for idx, pl in enumerate(hits):
            topic = html.escape(pl.get("source") or "Источник")
            snippet = preview_from_payload(pl, q)
            kb = InlineKeyboardMarkup(
                inline_keyboard=[[
                    InlineKeyboardButton(text="Показать полностью", callback_data=f"show_local:{idx}"),
                    InlineKeyboardButton(text="Скрыть", callback_data=f"hide_local:{idx}")
                ]]
            )
            await m.answer(f"<b>{topic}</b>\n{snippet}", parse_mode="HTML", reply_markup=kb)
        return

    # локально пусто: определим причину для unanswered
    reason = "no_local_hits"
    if (diag.get("rejected") and not diag.get("passed")):
        reason = "below_threshold"
    await asyncio.to_thread(log_unanswered, question_id, reason)

    # Этап 2 — сайт smp.edu.ru
    await msg.edit_text("Ищу на сайте smp.edu.ru…", parse_mode="HTML")
    try:
        site_text, site_results = await retrieve_site_live(q, max_results=5)
    except Exception as e:
        site_text, site_results = (f"Ошибка поиска на сайте: {e}", [])
    if site_results:
        await msg.edit_text("Нашёл ответы на сайте smp.edu.ru", parse_mode="HTML")
    else:
        await msg.edit_text("На сайте smp.edu.ru ничего не найдено", parse_mode="HTML")
    for chunk in _split_long(site_text):
        await m.answer(chunk, parse_mode="HTML")

    if site_results:
        await asyncio.to_thread(mark_answered, question_id, "site")
        return  # есть выдача на этапе 2 → веб не запускаем

    # Этап 3 — интернет (только если 1 и 2 пусто)
    await m.answer("Ищу в интернете…", parse_mode="HTML")
    await bot.send_chat_action(m.chat.id, ChatAction.TYPING)
    try:
        web_text, web_results = await retrieve_web_live(q, max_results=5)
    except Exception as e:
        web_text, web_results = (f"Ошибка веб-поиска: {e}", [])
    if web_results:
        await msg.edit_text("Нашёл ответы в интернете", parse_mode="HTML")
    else:
        await msg.edit_text("В интернете ничего не найдено", parse_mode="HTML")
    for chunk in _split_long(web_text):
        await m.answer(chunk, parse_mode="HTML")

    if web_results:
        await asyncio.to_thread(mark_answered, question_id, "web")


@router.callback_query(F.data.startswith("show_local:"))
async def accept_local(cb: CallbackQuery):
    try:
        idx = int(cb.data.split(":", 1)[1])
    except Exception:
        await cb.answer()
        return
    hits = pending_local.get(cb.from_user.id) or []
    if idx < 0 or idx >= len(hits):
        await cb.answer("Ответ не найден")
        return
    pl = hits[idx]
    msg_text, cites, files = await format_answer_from_payload(pl)
    await cb.message.edit_reply_markup()
    for chunk in _split_long(msg_text):
        await cb.message.answer(chunk, parse_mode="HTML")
    await cb.message.answer(
        "Дополнительную информацию вы можете прочитать в файлах, прикрепленных ниже.",
        parse_mode="HTML",
    )
    if files:
        await send_files(cb.message, files, limit=3)
    await cb.answer()
    pending_local.pop(cb.from_user.id, None)


@router.callback_query(F.data.startswith("hide_local:"))
async def reject_local(cb: CallbackQuery):
    try:
        idx = int(cb.data.split(":", 1)[1])
    except Exception:
        await cb.message.delete()
        await cb.answer()
        return
    hits = pending_local.get(cb.from_user.id)
    if hits and 0 <= idx < len(hits):
        hits[idx] = None
        if all(h is None for h in hits):
            pending_local.pop(cb.from_user.id, None)
    await cb.message.delete()
    await cb.answer()

# -------------------- запуск --------------------
async def main():
    init_db()  # создаём таблицы при старте
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
