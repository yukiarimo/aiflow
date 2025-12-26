import json
import os
import time

class ChatHistoryManager:
    def __init__(self, file_path='history.json'):
        self.file_path = file_path
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def load_history(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return []

    def save_history(self, history):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    def add_message(self, role, text, images=None):
        history = self.load_history()
        msg = {
            "id": f"msg-{int(time.time()*1000)}",
            "name": role,
            "text": text,
            "images": images or [],
            "timestamp": time.time()
        }
        history.append(msg)
        self.save_history(history)
        return msg

    def clear_history(self):
        self.save_history([])