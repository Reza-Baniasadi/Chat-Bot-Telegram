from telegram.ext import Updater, MessageHandler, Filters
from telegram import File, Bot
from telegram.utils.request import Request
from ultralytics import YOLO
import os
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
PROXY_URL = os.getenv("PROXY_URL", "socks5h://127.0.0.1:9050")

model = YOLO("yolov8n.pt")

def handle_photo(update, context):
    print("📸 عکس دریافت شد!")
    photo = update.message.photo[-1].get_file()
    file_path = "received.jpg"
    photo.download(file_path)

    results = model(file_path)
    result = results[0]

    classes = set([result.names[int(cls)] for cls in result.boxes.cls])

    if classes:
        update.message.reply_text("✅ در تصویر این اشیا شناسایی شدن:\n" + "\n".join(classes))
    else:
        update.message.reply_text("❌ هیچ شیئی شناسایی نشد.")

    os.remove(file_path)

def main():
    if USE_PROXY:
        req = Request(proxy_url=PROXY_URL)
        bot = Bot(token=TELEGRAM_TOKEN, request=req)
    else:
        bot = Bot(token=TELEGRAM_TOKEN)

    updater = Updater(bot=bot)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

