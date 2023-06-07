import os
dirs = os.listdir("D:\BaiduNetdiskDownload\Images\\train\\train")
with open("util/ovis_subset.txt", 'w') as f:
    f.writelines([dir + '\n' for dir in dirs])