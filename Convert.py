from moviepy.editor import *

filename = "D:/上課資料/IME/計畫/模擬影片/環繞飛行_上帝視角_快.3.23.mp4"
Exportname = "D:/上課資料/IME/計畫/模擬影片/環繞飛行_上帝視角_快.3.23.gif"

video = VideoFileClip(filename).subclip(00,14)
video.write_gif(Exportname, fps = 30)