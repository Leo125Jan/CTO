#!/usr/bin/python3

from moviepy.editor import *

filename = "/home/leo/影片/Video/Plan_Video/5.16/Zoom_in_out.mp4"
Exportname = "/home/leo/影片/Video/Plan_Video/5.16/Zoom_in_out.gif"

video = VideoFileClip(filename).subclip(00,27)
video.write_gif(Exportname, fps = 30)