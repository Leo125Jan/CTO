from moviepy.editor import *

filename = "D:/Leo/IME/Paper Study/Coverage Control/Quality based switch mode/Data/Switch Mode.3.1.mp4"
Exportname = "D:/Leo/IME/Paper Study/Coverage Control/Quality based switch mode/Data/Switch_Mode.3.1.gif"

video = VideoFileClip(filename).subclip(00,21)
video.write_gif(Exportname, fps = 20)