import os
import fnmatch

def find_tar_files(directory='./'):
    tar_files = []
    for root, dirs, files in os.walk(directory):
        for file in fnmatch.filter(files, '*.tar'):
            tar_files.append(os.path.join(root, file))
    return tar_files

