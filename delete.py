import os


root = '/home/cristina/Desktop/mmaction2/data/ucf101/videos_256p_dense_cache/FloorGymnastics'



for filename in os.listdir(root):
    if filename.endswith(".mp4.mp4"):
        os.remove(filename)