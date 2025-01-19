from telethon import TelegramClient, events
import os
from dotenv import load_dotenv

class RealTimeTelegramScraper:
    def __init__(self):
        load_dotenv('.env')
        self.api_id = os.getenv('TG_API_ID')
        self.api_hash = os.getenv('TG_API_HASH')
        self.client = TelegramClient('real_time_session', self.api_id, self.api_hash)
        self.data = []

    async def message_handler(self, event):
        message = event.message
        if message.text and self.is_amharic(message.text):
            self.data.append({
                'Channel Username': event.chat.username,
                'ID': message.id,
                'Message': message.text,
                'Date': message.date,
                'Media Path': None  # Add media handling if needed
            })
            print(f"New message from {event.chat.username}: {message.text}")

    @staticmethod
    def is_amharic(text):
        return bool(re.search(r'[\u1200-\u137F]', text))  # Amharic Unicode range

    async def run(self, channels):
        await self.client.start()
        for channel in channels:
            await self.client.get_entity(channel)  # Ensure the client joins the channel
        self.client.add_event_handler(self.message_handler, events.NewMessage(chats=channels))

        print("Listening for new messages...")
        await self.client.run_until_disconnected()

    def get_data(self):
        return self.data
