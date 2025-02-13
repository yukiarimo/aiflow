import json
import os
from cryptography.fernet import Fernet, InvalidToken
from aiflow import agi

class ChatHistoryManager:
    def __init__(self, config, use_file=False):
        self.config = config
        self.fernet = Fernet(self._get_encryption_key())
        self.use_file = use_file
        self.memory_storage = {}  # Format: {username: {chat_id: history}}

    def _get_user_folder(self, username):
        path = os.path.join("db/history", username)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_file_path(self, username, chat_id):
        return os.path.join(self._get_user_folder(username), str(chat_id))

    def _get_encryption_key(self):
        key = self.config['security'].get('encryption_key')
        if not key:
            key = Fernet.generate_key().decode()
            self.config['security']['encryption_key'] = key
            agi.get_config(self.config)
        return key.encode()

    def _read_encrypted_file(self, path):
        try:
            with open(path, 'rb') as file:
                return self.fernet.decrypt(file.read()).decode()
        except (FileNotFoundError, InvalidToken):
            return None

    def _write_encrypted_file(self, path, data):
        with open(path, 'wb') as file:
            file.write(self.fernet.encrypt(data.encode()))

    def _get_memory_storage(self, username):
        if username not in self.memory_storage:
            self.memory_storage[username] = {}
        return self.memory_storage[username]

    def create_chat_history_file(self, username, chat_id):
        template = [
            {
                "name": self.config['ai']['names'][0],
                "text": "Hello",
                "data": None,
                "type": "text",
            },
            {
                "name": self.config['ai']['names'][1],
                "text": "Hello",
                "data": None,
                "type": "text",
            },
        ]
        
        if self.use_file:
            path = self._get_file_path(username, chat_id)
            self._write_encrypted_file(path, json.dumps(template))
        else:
            storage = self._get_memory_storage(username)
            storage[str(chat_id)] = template

    def save_chat_history(self, chat_history, username, chat_id):
        if self.use_file:
            path = self._get_file_path(username, chat_id)
            self._write_encrypted_file(path, json.dumps(chat_history))
        else:
            storage = self._get_memory_storage(username)
            storage[str(chat_id)] = chat_history

    def load_chat_history(self, username, chat_id):
        if self.use_file:
            path = self._get_file_path(username, chat_id)
            decrypted = self._read_encrypted_file(path)
            if decrypted is None:
                self.create_chat_history_file(username, chat_id)
                print(f"Chat history file for {chat_id} does not exist, creating a new one for {username}")
                return []
            return json.loads(decrypted)
        else:
            storage = self._get_memory_storage(username)
            if str(chat_id) not in storage:
                self.create_chat_history_file(username, chat_id)
                print(f"Chat history for {chat_id} does not exist, creating a new one for {username}")
                return []
            return storage[str(chat_id)]

    def delete_chat_history_file(self, username, chat_id):
        if self.use_file:
            path = self._get_file_path(username, chat_id)
            os.remove(path) if os.path.exists(path) else None
        else:
            storage = self._get_memory_storage(username)
            storage.pop(str(chat_id), None)

    def rename_chat_history_file(self, username, old_chat_id, new_chat_id):
        if self.use_file:
            old_path = self._get_file_path(username, old_chat_id)
            new_path = self._get_file_path(username, new_chat_id)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
        else:
            storage = self._get_memory_storage(username)
            if str(old_chat_id) in storage:
                storage[str(new_chat_id)] = storage.pop(str(old_chat_id))

    def list_history_files(self, username):
        if self.use_file:
            folder = self._get_user_folder(username)
            return sorted(
                [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))],
                key=lambda x: x.lower()
            )
        else:
            storage = self._get_memory_storage(username)
            return sorted(storage.keys(), key=lambda x: x.lower())

    def delete_message(self, username, chat_id, target_message):
        history = self.load_chat_history(username, chat_id)
        indices_to_delete = [i for i, msg in enumerate(history)
                           if msg['message'].strip() == target_message.strip()]

        for index in sorted(indices_to_delete, reverse=True):
            del history[index]
            if index < len(history):
                del history[index]

        self.save_chat_history(history, username, chat_id)