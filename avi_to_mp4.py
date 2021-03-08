import os

## get list of all videos

root = '/home/cristina/Desktop/mmaction2/data/ucf101/videos_256p_dense_cache'
all_avis= []

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True


for path, subdirs, files in os.walk(root):
    for name in files:
        all_avis.append(os.path.join(path, name))


for avi_file in all_avis:
    output_name= avi_file[:-4]+'.mp4'
    convert_avi_to_mp4(avi_file, output_name)