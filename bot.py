# bot.py
import os
import sys
import asyncio
import shutil
import subprocess
import tempfile
from typing import List
import re
import logging

from config import settings, setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# --- sys.path safety ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- aiogram ---
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message, ReplyKeyboardMarkup, KeyboardButton, FSInputFile,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery,
)
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

# --- our modules ---
from retrieval_local import retrieve_local
from ingest.pdf_ingest import ingest_pdf           # админ: одиночный PDF
from ingest.ingest_generic import ingest_path      # индексация папки (teach/)
from store_qdrant import get_client                # /clear_index
from db_local import (
    init_db, upsert_user, insert_question, mark_answered,
    log_unanswered, log_answer_score, fetch_unanswered,
    set_last_question, get_last_question,
    set_pending_files, pop_pending_files,
    log_feedback, update_feedback_comment, fetch_feedback,
    get_user_id,
)

# Этапы 2/3 (поиск по сайту / по вебу) — должны быть в проекте
from retrieve_site_live import retrieve_site_live
from retrieve_web_live  import retrieve_web_live

# -------------------- настройки --------------------
if not settings.BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не задан в .env")

ADMIN_IDS = {int(x) for x in settings.ADMIN_IDS.split(",") if x.strip().isdigit()}
QDRANT_COLLECTION = settings.QDRANT_COLLECTION
TEACH_DIR = settings.TEACH_DIR

bot = Bot(settings.BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

# -------------------- форматирование ответа --------------------
_DIGITS = {
    "0": "0️⃣",
    "1": "1️⃣",
    "2": "2️⃣",
    "3": "3️⃣",
    "4": "4️⃣",
    "5": "5️⃣",
    "6": "6️⃣",
    "7": "7️⃣",
    "8": "8️⃣",
    "9": "9️⃣",
}

def format_text(text: str) -> str:
    """Заменяет *звёздочки* на <b> и цифры в начале строк на смайлики."""
    if not text:
        return ""
    # *text* → <b>text</b>
    text = re.sub(r"\*(.+?)\*", r"<b>\1</b>", text)

    def _emojify_line(line: str) -> str:
        if re.match(r"^\d+[\.)]", line.strip()):
            return "".join(_DIGITS.get(ch, ch) for ch in line)
        return line

    lines = text.splitlines()
    lines = [_emojify_line(ln) for ln in lines]
    return "\n".join(lines)


# -------------------- вспомогалки UI/админ --------------------
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

def admin_kb():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="ℹ️ Помощь"), KeyboardButton(text="➕ Добавить документ")],
        ],
        resize_keyboard=True
    )

def user_kb():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="ℹ️ Помощь")]], resize_keyboard=True
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


def _docx_to_pdf_sync(src_path: str) -> str | None:
    if not os.path.isfile(src_path):
        return None
    soffice = shutil.which("soffice")
    if not soffice:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="docx2pdf_") as tmpdir:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf", "--outdir", tmpdir, src_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            base = os.path.splitext(os.path.basename(src_path))[0] + ".pdf"
            tmp_pdf = os.path.join(tmpdir, base)
            if not os.path.exists(tmp_pdf):
                return None
            dst_pdf = os.path.join(CONVERT_DIR, base)
            shutil.move(tmp_pdf, dst_pdf)
            return dst_pdf
    except Exception as e:
        logger.exception("docx_to_pdf failed: %s", e)
        return None


async def docx_to_pdf(src_path: str, timeout: int = 20) -> str | None:
    try:
        return await asyncio.wait_for(asyncio.to_thread(_docx_to_pdf_sync, src_path), timeout)
    except Exception:
        logger.exception("docx_to_pdf timeout or error")
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
                pdf_path = await docx_to_pdf(p)
                if pdf_path and os.path.exists(pdf_path):
                    to_send_path = pdf_path
                    caption = os.path.basename(pdf_path)
                else:
                    await m.answer(
                        "Не удалось преобразовать DOCX в PDF, отправляю исходный файл.",
                        parse_mode="HTML",
                    )

            await m.answer_document(FSInputFile(to_send_path), caption=caption)
            count += 1
        except Exception:
            logger.exception("send_files error for %s", p)
            await m.answer(f"Не удалось отправить файл: {p}", parse_mode="HTML")


# клавиатура оценки ответа
rate_kb = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="Плюс", callback_data="rate_plus"),
            InlineKeyboardButton(text="Минус", callback_data="rate_minus"),
        ]
    ]
)

HELP_TEXT = (
    "Я могу отвечать на вопросы, ищя информацию в локальной базе, на сайте и в интернете.\n"
    "Основные команды:\n"
    "/start — начать работу\n"
    "/help — показать эту справку"
)


@router.callback_query(F.data == "get_doc")
async def cb_get_doc(c: CallbackQuery):
    db_uid = await asyncio.to_thread(get_user_id, c.from_user.id)
    files = await asyncio.to_thread(pop_pending_files, db_uid) if db_uid else []
    if files:
        await send_files(c.message, files)
    else:
        await c.message.answer("Документ не найден.", parse_mode="HTML")
    await c.answer()

# -------------------- FSM: добавление документа (PDF) --------------------
class AddDoc(StatesGroup):
    waiting_title = State()
    waiting_file = State()


class Feedback(StatesGroup):
    waiting_text = State()

@router.message(F.text == "➕ Добавить документ")
async def add_doc(m: Message, state: FSMContext):
    if not is_admin(m.from_user.id):
        return await m.answer("Эта функция доступна только администратору.", parse_mode="HTML")
    await state.set_state(AddDoc.waiting_title)
    await m.answer("Введите название документа (как показывать пользователям).", parse_mode="HTML")

@router.message(AddDoc.waiting_title, F.text)
async def got_title(m: Message, state: FSMContext):
    await state.update_data(title=m.text.strip())
    await state.set_state(AddDoc.waiting_file)
    await m.answer("Пришлите PDF-файл документом (не фото).", parse_mode="HTML")

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


# -------------------- обратная связь --------------------
async def send_rating(m: Message):
    await m.answer("Оцените ответ помощника", reply_markup=rate_kb, parse_mode="HTML")


@router.callback_query(F.data == "rate_plus")
async def rate_plus(c: CallbackQuery):
    db_uid = await asyncio.to_thread(get_user_id, c.from_user.id)
    qid, _ = await asyncio.to_thread(get_last_question, db_uid) if db_uid else (None, None)
    if db_uid:
        await asyncio.to_thread(log_feedback, qid, db_uid, 1, None)
    await c.message.edit_reply_markup()
    await c.message.answer("Благодарим вас за ответ", parse_mode="HTML")
    await c.answer()


@router.callback_query(F.data == "rate_minus")
async def rate_minus(c: CallbackQuery, state: FSMContext):
    db_uid = await asyncio.to_thread(get_user_id, c.from_user.id)
    qid, _ = await asyncio.to_thread(get_last_question, db_uid) if db_uid else (None, None)
    fid = None
    if db_uid:
        fid = await asyncio.to_thread(log_feedback, qid, db_uid, -1, None)
    await state.update_data(feedback_id=fid)
    await c.message.edit_reply_markup()
    await c.message.answer("Благодарим вас за ответ", parse_mode="HTML")
    await c.message.answer("Опишите пожалуйста свою проблему", parse_mode="HTML")
    await state.set_state(Feedback.waiting_text)
    await c.answer()


@router.message(Feedback.waiting_text, F.text)
async def feedback_text(m: Message, state: FSMContext):
    data = await state.get_data()
    feedback_id = data.get("feedback_id")
    problem = m.text.strip()
    db_uid = await asyncio.to_thread(get_user_id, m.from_user.id)
    qid, question = await asyncio.to_thread(get_last_question, db_uid) if db_uid else (None, None)
    if feedback_id:
        await asyncio.to_thread(update_feedback_comment, feedback_id, problem)
    user = m.from_user
    info = (
        "Негативная оценка\n"
        f"Ник: @{user.username or 'не указан'}\n"
        f"Телефон: не указан\n"
        f"ID: {user.id}\n"
        f"Вопрос: {question or ''}\n"
        f"Проблема: {problem}"
    )
    for admin_id in ADMIN_IDS:
        await bot.send_message(admin_id, info, parse_mode="HTML")
    await m.answer("Благодарим вас за ответ", parse_mode="HTML")
    await state.clear()


@router.message(Command("help"))
async def cmd_help(m: Message):
    await m.answer(HELP_TEXT, parse_mode="HTML")


@router.message(F.text == "ℹ️ Помощь")
async def help_button(m: Message):
    await m.answer(HELP_TEXT, parse_mode="HTML")

# -------------------- команды админа --------------------
@router.message(CommandStart())
async def cmd_start(m: Message):
    if is_admin(m.from_user.id):
        await m.answer(
            "Здравствуйте! Отправьте вопрос — я пришлю найденные фрагменты текстом и приложу исходные файлы.\n"
            "Команды:\n"
            "• /clear_index — очистить локальный индекс Qdrant\n"
            "• /ingest_teach — проиндексировать все файлы из папки teach/ (source_group=teach)",
            reply_markup=admin_kb(), parse_mode="HTML"
        )
    else:
        await m.answer(
            "Здравствуйте! Отправьте свой вопрос — я пришлю найденные фрагменты из локальной базы и приложу файлы.",
            reply_markup=user_kb(), parse_mode="HTML"
        )

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


@router.message(Command("unanswered"))
async def cmd_unanswered(m: Message):
    if not is_admin(m.from_user.id):
        return await m.answer("Эта команда доступна только администратору.", parse_mode="HTML")
    parts = m.text.split(maxsplit=1)
    limit = 10
    if len(parts) == 2:
        try:
            limit = int(parts[1])
        except ValueError:
            return await m.answer("Использование: /unanswered N", parse_mode="HTML")
    rows = await asyncio.to_thread(fetch_unanswered, limit)
    if not rows:
        return await m.answer("Список пуст.", parse_mode="HTML")
    lines = [f"{i+1}. {r['question']} — {r['reason']}" for i, r in enumerate(rows)]
    text = "\n".join(lines)
    for chunk in _split_long(text):
        await m.answer(chunk, parse_mode="HTML")


@router.message(Command("feedback"))
async def cmd_feedback(m: Message):
    if not is_admin(m.from_user.id):
        return await m.answer("Эта команда доступна только администратору.", parse_mode="HTML")
    parts = m.text.split(maxsplit=1)
    limit = 10
    if len(parts) == 2:
        try:
            limit = int(parts[1])
        except ValueError:
            return await m.answer("Использование: /feedback N", parse_mode="HTML")
    rows = await asyncio.to_thread(fetch_feedback, limit)
    if not rows:
        return await m.answer("Нет отзывов.", parse_mode="HTML")
    lines = []
    for r in rows:
        sign = "+" if (r["rating"] or 0) > 0 else "-"
        lines.append(f"{r['created_at']}: {sign} {r['question'] or ''} {r['comment'] or ''}")
    text = "\n".join(lines)
    for chunk in _split_long(text):
        await m.answer(chunk, parse_mode="HTML")

# -------------------- обработка вопроса пользователя --------------------
@router.message(F.text & ~F.text.in_({"➕ Добавить документ"}))
async def handle_question(m: Message, state: FSMContext):
    q = m.text.strip()
    logger.info("[handle_question] question=%s", q)

    tg = m.from_user
    user_id = await asyncio.to_thread(upsert_user, tg.id, tg.username, tg.first_name, tg.last_name)
    question_id = await asyncio.to_thread(insert_question, user_id, q)
    await asyncio.to_thread(set_last_question, user_id, question_id, q)
    logger.debug("Stored question %s from user %s", question_id, tg.id)

    start = asyncio.get_event_loop().time()
    await m.answer("Ищу по локальной базе…", parse_mode="HTML")
    msg_text, cites, files, diag = await retrieve_local(q, top_k=3, prefer_spravochnik=False)
    elapsed = asyncio.get_event_loop().time() - start
    logger.info("local search finished in %.2fs", elapsed)

    for payload, score in (diag.get("passed") or []):
        await asyncio.to_thread(log_answer_score, question_id, payload, score, True)
    for payload, score in (diag.get("rejected") or []):
        await asyncio.to_thread(log_answer_score, question_id, payload, score, False)

    found_local = bool(cites)
    msg_text = format_text(msg_text)
    for chunk in _split_long(msg_text):
        await m.answer(chunk, parse_mode="HTML")
    logger.debug("found_local=%s", found_local)

    if found_local:
        await asyncio.to_thread(mark_answered, question_id, "local")
        if files:
            await asyncio.to_thread(set_pending_files, user_id, files)
            kb = InlineKeyboardMarkup(
                inline_keyboard=[[InlineKeyboardButton(text="Получить документ", callback_data="get_doc")]]
            )
            await m.answer("Нажмите кнопку, чтобы получить документ.", reply_markup=kb, parse_mode="HTML")
        await send_rating(m)
        return

    # локально пусто: определим причину для unanswered
    reason = "no_local_hits"
    if (diag.get("rejected") and not diag.get("passed")):
        reason = "below_threshold"
    logger.info("no local results, reason=%s", reason)
    await asyncio.to_thread(log_unanswered, question_id, reason)

    logger.info("Step 2: поиск на сайте smp.edu.ru")
    await m.answer("Ищу на сайте smp.edu.ru…", parse_mode="HTML")
    try:
        site_text, site_results = await retrieve_site_live(q, max_results=5)
    except Exception as e:
        logger.exception("site search error")
        site_text, site_results = (f"Ошибка поиска на сайте: {e}", [])
    site_text = format_text(site_text)
    for chunk in _split_long(site_text):
        await m.answer(chunk, parse_mode="HTML")
    logger.debug("site_results=%s", site_results)

    if site_results:
        await asyncio.to_thread(mark_answered, question_id, "site")
        await send_rating(m)
        return

    logger.info("Step 3: поиск в интернете")
    await m.answer("Ищу в интернете…", parse_mode="HTML")
    try:
        web_text, web_results = await retrieve_web_live(q, max_results=5)
    except Exception as e:
        logger.exception("web search error")
        web_text, web_results = (f"Ошибка веб-поиска: {e}", [])
    web_text = format_text(web_text)
    for chunk in _split_long(web_text):
        await m.answer(chunk, parse_mode="HTML")
    logger.debug("web_results=%s", web_results)

    if web_results:
        await asyncio.to_thread(mark_answered, question_id, "web")
        logger.info("Answered via web search")
    await send_rating(m)

# -------------------- запуск --------------------
async def main():
    init_db()  # создаём таблицы при старте
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
