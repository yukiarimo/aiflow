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

    def _get_encryption_key(self):
        key = self.config['security'].get('encryption_key')
        if not key:
            key = Fernet.generate_key().decode()
            self.config['security']['encryption_key'] = key
            agi.get_config(self.config)
        return key.encode()

    def _get_user_folder(self, username):
        path = os.path.join("db/history", username)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_file_path(self, username, chat_id):
        return os.path.join(self._get_user_folder(username), str(chat_id))

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
        return self.memory_storage.setdefault(username, {})

    def _get_template(self):
        return [
            {"name": self.config['ai']['names'][0], "text": "Hello", "data": None, "type": "text", "id": "msg-0"},
            {"name": self.config['ai']['names'][1], "text": "Hi", "data": None, "type": "text", "id": "msg-1"},
            {"name": self.config['ai']['names'][0], "text": "How are you?", "data": None, "type": "text", "id": "msg-2"},
            {"name": self.config['ai']['names'][1], "text": "I'm fine, thanks!", "data": None, "type": "text", "id": "msg-3"}
        ]

    def create_chat_history_file(self, username, chat_id):
        template = self._get_template()
        if self.use_file:
            self._write_encrypted_file(self._get_file_path(username, chat_id), json.dumps(template))
        else:
            self._get_memory_storage(username)[str(chat_id)] = template

    def save_chat_history(self, chat_history, username, chat_id):
        if self.use_file:
            self._write_encrypted_file(self._get_file_path(username, chat_id), json.dumps(chat_history))
        else:
            self._get_memory_storage(username)[str(chat_id)] = chat_history

    def load_chat_history(self, username, chat_id):
        if self.use_file:
            decrypted = self._read_encrypted_file(self._get_file_path(username, chat_id))
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
            if os.path.exists(path):
                os.remove(path)
        else:
            self._get_memory_storage(username).pop(str(chat_id), None)

    def rename_chat_history_file(self, username, old_chat_id, new_chat_id):
        if self.use_file:
            old_path = self._get_file_path(username, old_chat_id)
            if os.path.exists(old_path):
                os.rename(old_path, self._get_file_path(username, new_chat_id))
        else:
            storage = self._get_memory_storage(username)
            if str(old_chat_id) in storage:
                storage[str(new_chat_id)] = storage.pop(str(old_chat_id))

    def list_history_files(self, username):
        if self.use_file:
            folder = self._get_user_folder(username)
            return sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))], key=str.lower)
        else:
            return sorted(self._get_memory_storage(username).keys(), key=str.lower)

    def delete_message(self, user_id, chat_id, message_id):
        """Delete a message by its ID."""
        history = self.load_chat_history(user_id, chat_id)
        # Find and remove the message with matching ID
        for i, msg in enumerate(history):
            if msg.get('id') == message_id:
                history.pop(i)
                break
        return self.save_chat_history(history, user_id, chat_id)

    def edit_message(self, user_id, chat_id, message_id, new_text):
        """Edit a message by its ID."""
        history = self.load_chat_history(user_id, chat_id)
        # Find and edit the message with matching ID
        for msg in history:
            if msg.get('id') == message_id:
                msg['text'] = new_text
                break
        return self.save_chat_history(history, user_id, chat_id)