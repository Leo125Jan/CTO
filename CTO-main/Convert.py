#!/usr/bin/python3

from moviepy.editor import *

filename = "C:/Users/leomd/IME/Paper Study/Coverage Control/Quality based switch mode/Data/2024/5.27/Video/W_CBAA_Edit.mp4"
Exportname = "C:/Users/leomd/IME/Paper Study/Coverage Control/Quality based switch mode/Data/2024/5.27/Video/W_CBAA_Edit.gif"

video = VideoFileClip(filename).subclip(00,15)
video.write_gif(Exportname, fps = 30)