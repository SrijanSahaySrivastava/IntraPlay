import tkinter as tk
import tkinter.filedialog as filedialog
import vlc
import datetime

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")
        self.root.geometry("800x600")

        # VLC player
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.player.set_hwnd(self.root.winfo_id())

        # Video player frame
        self.player_frame = tk.Frame(self.root)
        self.player_frame.pack()

        # Browse button
        self.browse_button = tk.Button(self.root, text="Browse", command=self.browse_video)
        self.browse_button.pack()

        # Time bar window
        self.time_window = None
        self.time_bar = None
        self.time_label = None

        # Bind keys for volume control and video seeking
        self.root.bind("<Up>", self.volume_up)
        self.root.bind("<Down>", self.volume_down)
        self.root.bind("<Left>", self.seek_backward)
        self.root.bind("<Right>", self.seek_forward)

        # Bind Esc key to close the program
        self.root.bind("<Escape>", self.close_program)

    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.avi")])
        if file_path:
            self.play_video(file_path)

    def play_video(self, file_path):
        media = self.instance.media_new(file_path)
        self.player.set_media(media)
        self.player.play()
        self.update_time()

    def update_time(self):
        if self.time_window is None:
            self.time_window = tk.Toplevel(self.root)
            self.time_window.title("Time Bar")
            self.time_window.geometry("400x50")
            self.time_bar = tk.Scale(self.time_window, from_=0, to= 0, orient=tk.HORIZONTAL)
            self.time_bar.pack(fill=tk.X)
            self.time_label = tk.Label(self.time_window)
            self.time_label.pack()

        current_time = self.player.get_time()
        total_time = self.player.get_length()

        if total_time > 0:
            progress = current_time
            self.time_bar["to"] = total_time
            self.time_bar.set(progress)

            remaining_time = datetime.timedelta(milliseconds=(total_time - current_time))
            elapsed_time = datetime.timedelta(milliseconds=current_time)
            time_label = f"Elapsed: {elapsed_time} | Remaining: {remaining_time}"
            self.time_label.config(text=time_label)

        self.time_window.after(1000, self.update_time)

    def volume_up(self, event):
        current_volume = self.player.audio_get_volume()
        if current_volume < 100:
            new_volume = min(current_volume + 10, 100)
            self.player.audio_set_volume(new_volume)

    def volume_down(self, event):
        current_volume = self.player.audio_get_volume()
        if current_volume > 0:
            new_volume = max(current_volume - 10, 0)
            self.player.audio_set_volume(new_volume)

    def seek_backward(self, event):
        current_time = self.player.get_time()
        new_time = max(current_time - 5000, 0)
        self.player.set_time(new_time)

    def seek_forward(self, event):
        current_time = self.player.get_time()
        total_time = self.player.get_length()
        new_time = min(current_time + 5000, total_time)
        self.player.set_time(new_time)

    def close_program(self, event):
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()
