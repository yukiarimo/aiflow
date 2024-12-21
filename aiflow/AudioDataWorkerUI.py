import os
import tkinter as tk
from tkinter import messagebox, scrolledtext
import pygame
import re
from plyer import notification
pygame.mixer.init()

class TranscriptEditor:
    def __init__(self, master, metadata_file="metadata.txt"):
        self.master = master
        master.title("Transcript Editor")
        master.geometry("900x650")

        # Load metadata
        self.metadata_file = metadata_file
        self.entries = self.load_metadata()

        # Sort entries by chunk number
        self.sort_entries()

        # Initialize filtered entries
        self.filtered_entries = self.entries.copy()

        # Selected entry index
        self.selected_index = None

        # Setup UI
        self.setup_ui()

    def load_metadata(self):
        entries = []
        if not os.path.exists(self.metadata_file):
            messagebox.showerror("Error", f"{self.metadata_file} not found.")
            return entries
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    audio_path = parts[0]
                    if os.path.exists(audio_path):
                        try:
                            sound = pygame.mixer.Sound(audio_path)
                            duration = sound.get_length()  # Duration in seconds
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to get duration for {audio_path}: {e}")
                            duration = 0
                    else:
                        duration = 0
                    entries.append({
                        "filepath": parts[0],
                        "speaker": parts[1],
                        "language": parts[2],
                        "transcript": parts[3],
                        "duration": duration
                    })
        return entries

    def sort_entries(self):
        # Extract chunk number and sort
        def get_chunk_number(entry):
            match = re.search(r'chunk_(\d+)\.wav', os.path.basename(entry["filepath"]))
            if match:
                return int(match.group(1))
            return 0  # Default if not matched

        self.entries.sort(key=get_chunk_number)

    def save_metadata(self):
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            for entry in self.entries:
                line = f'{entry["filepath"]}|{entry["speaker"]}|{entry["language"]}|{entry["transcript"]}\n'
                f.write(line)

    def setup_ui(self):
        # Top frame for filters
        top_frame = tk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        lbl_filter = tk.Label(top_frame, text="Filter by Duration (seconds):", font=("Helvetica", 12))
        lbl_filter.pack(side=tk.LEFT, padx=(0, 10))

        lbl_min = tk.Label(top_frame, text="Min:", font=("Helvetica", 12))
        lbl_min.pack(side=tk.LEFT)
        self.entry_min = tk.Entry(top_frame, width=10)
        self.entry_min.pack(side=tk.LEFT, padx=(0, 10))

        lbl_max = tk.Label(top_frame, text="Max:", font=("Helvetica", 12))
        lbl_max.pack(side=tk.LEFT)
        self.entry_max = tk.Entry(top_frame, width=10)
        self.entry_max.pack(side=tk.LEFT, padx=(0, 10))

        btn_filter = tk.Button(top_frame, text="Apply Filter", command=self.apply_filter)
        btn_filter.pack(side=tk.LEFT, padx=(10, 0))

        btn_clear = tk.Button(top_frame, text="Clear Filter", command=self.clear_filter)
        btn_clear.pack(side=tk.LEFT, padx=(5, 0))

        # Main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left frame for listbox
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        lbl = tk.Label(left_frame, text="Audio Files", font=("Helvetica", 14))
        lbl.pack(pady=5)

        self.listbox = tk.Listbox(left_frame, width=40)
        self.listbox.pack(fill=tk.Y, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

        # Populate listbox
        self.populate_listbox()

        # Right frame for controls
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # Play button
        self.play_button = tk.Button(right_frame, text="Play", command=self.play_audio, state=tk.DISABLED, width=15)
        self.play_button.pack(pady=5)

        # Duration label
        self.lbl_duration = tk.Label(right_frame, text="Duration: 0.0 seconds", font=("Helvetica", 12))
        self.lbl_duration.pack(pady=5)

        # Transcript editor
        lbl_transcript = tk.Label(right_frame, text="Transcript:", font=("Helvetica", 12))
        lbl_transcript.pack(anchor='w')

        self.transcript_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=25)
        self.transcript_text.pack(fill=tk.BOTH, expand=True)
        self.transcript_text.config(state=tk.DISABLED)
        self.transcript_text.bind("<KeyRelease>", self.on_text_change)

        # Delete button
        self.delete_button = tk.Button(right_frame, text="Delete", command=self.delete_entry, state=tk.DISABLED, fg='red', width=15)
        self.delete_button.pack(pady=5)

    def populate_listbox(self):
        self.listbox.delete(0, tk.END)
        for idx, entry in enumerate(self.filtered_entries):
            display_text = f"{os.path.basename(entry['filepath'])} ({entry['duration']:.2f}s)"
            self.listbox.insert(tk.END, display_text)

    def apply_filter(self):
        min_duration = self.entry_min.get().strip()
        max_duration = self.entry_max.get().strip()

        try:
            min_dur = float(min_duration) if min_duration else 0
            max_dur = float(max_duration) if max_duration else float('inf')
            if min_dur < 0 or max_dur < 0:
                raise ValueError
            if min_dur > max_dur:
                messagebox.showerror("Error", "Minimum duration cannot be greater than maximum duration.")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values for duration.")
            return

        self.filtered_entries = [
            entry for entry in self.entries
            if min_dur <= entry["duration"] <= max_dur
        ]
        self.populate_listbox()
        self.transcript_text.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.delete_button.config(state=tk.DISABLED)
        self.selected_index = None

    def clear_filter(self):
        self.entry_min.delete(0, tk.END)
        self.entry_max.delete(0, tk.END)
        self.filtered_entries = self.entries.copy()
        self.populate_listbox()
        self.transcript_text.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.delete_button.config(state=tk.DISABLED)
        self.selected_index = None

    def on_select(self, event):
        if not self.listbox.curselection():
            return
        index = self.listbox.curselection()[0]
        self.selected_index = index
        entry = self.filtered_entries[index]

        # Enable buttons
        self.play_button.config(state=tk.NORMAL)
        self.delete_button.config(state=tk.NORMAL)

        # Display duration
        self.lbl_duration.config(text=f"Duration: {entry['duration']:.2f} seconds")

        # Display transcript
        self.transcript_text.config(state=tk.NORMAL)
        self.transcript_text.delete(1.0, tk.END)
        self.transcript_text.insert(tk.END, entry["transcript"])
        self.transcript_text.config(state=tk.NORMAL)

    def play_audio(self):
        if self.selected_index is None:
            return
        entry = self.filtered_entries[self.selected_index]
        audio_path = entry["filepath"]
        if not os.path.exists(audio_path):
            messagebox.showerror("Error", f"Audio file {audio_path} not found.")
            return
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {e}")

    def on_text_change(self, event):
        if self.selected_index is None:
            return
        new_text = self.transcript_text.get(1.0, tk.END).strip()
        # Update both filtered_entries and entries
        filtered_entry = self.filtered_entries[self.selected_index]
        filtered_entry["transcript"] = new_text
        # Find and update the main entries list
        for entry in self.entries:
            if entry["filepath"] == filtered_entry["filepath"]:
                entry["transcript"] = new_text
                break
        self.save_metadata()

    def delete_entry(self):
        if self.selected_index is None:
            return
        entry = self.filtered_entries[self.selected_index]
        
        # Delete audio file
        try:
            os.remove(entry["filepath"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete audio file: {e}")
            return
        
        # Remove from entries and filtered_entries
        self.entries = [e for e in self.entries if e["filepath"] != entry["filepath"]]
        del self.filtered_entries[self.selected_index]
        self.listbox.delete(self.selected_index)
        self.save_metadata()
        
        # Clear transcript
        self.transcript_text.config(state=tk.DISABLED)
        self.transcript_text.delete(1.0, tk.END)
        self.lbl_duration.config(text="Duration: 0.0 seconds")
        
        # Disable buttons
        self.play_button.config(state=tk.DISABLED)
        self.delete_button.config(state=tk.DISABLED)
        self.selected_index = None
        
        # Show system notification
        notification.notify(
            title="Deleted",
            message="Audio file and transcript have been deleted.",
            app_name="YourAppName",
            timeout=5
        )

def main(metadata_file="metadata.txt"):
    if not os.path.exists(metadata_file):
        messagebox.showerror("Error", f"{metadata_file} not found.")
        return
    root = tk.Tk()
    app = TranscriptEditor(root, metadata_file)
    root.mainloop()