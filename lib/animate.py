# -*- coding: utf-8 -*-
"""
make gif from select folder
credit to https://github.com/dm20/gif-maker
"""
# from tkinter import filedialog
import imageio
import os
import sys
import re
import shutil
import numpy as np
from os.path import isfile, join

from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[1])
if __name__ == '__main__' and __package__ is None:
    __package__ = 'kurosc'

from lib.plotformat import PlotSetup


"""
gif_maker many thanks
https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
fp_in = "/path/to/image_*.png"
fp_out = "/path/to/image.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)
"""


class Animate(object):
    def __init__(self, out_folder_name:str):
        self.plot_directory = out_folder_name
        self.img_name = None
        self.fmt = PlotSetup('animation')

    def to_gif(self,
               targetpath: str = None,
               delay: float = 1.33,
               sort: bool = False,
               clean: bool = False,
               ext: str = 'png',
               ):
        if not targetpath:
            targetpath = self.plot_directory
        filelist = [f for f in os.listdir(targetpath)
                    if isfile(join(targetpath, f))
                    and f.endswith(ext)]

        if sort:
            # index = lambda x: re.search('\.\d_',str(x)).group()
            s = lambda x: re.split(r'\d*\.*\d*_',str(x),1)   # t = 1.4_20200505... & 15_2020..
            index = np.array([s(file) for file in filelist],dtype=str)
            if len(index.shape)==1:
                print('err 1D arry')
                return False

            files = np.array([index[...,-1],filelist],dtype=str).T
            files = files[files[...,0].argsort()]
            filelist = list(files[:,1])
            # print(filelist)
        img = lambda f: imageio.imread(os.path.join(targetpath, f))
        images = list(map(img, filelist))

        # print('***',targetpath.name)
        self.img_name = self.fmt.plot_name(targetpath,'gif')
        # print(self.img_name)
        imageio.mimsave(self.img_name, images, duration=delay)

        if clean:
            self.cleanup(targetpath)

    def cleanup(self, targetpath:str = None):

        for filename in os.listdir(targetpath):
            file_path = os.path.join(targetpath, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)

                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
                return False

        try:
            os.rmdir(targetpath)

        except OSError as err:
            print(err)
            return False
        return True


# TODO enable zip img
    #     try:
    #         new_fldr = self.fmt.plot_name(str(targetpath.stem)).stem
    #         archive = self.plot_directory/new_fldr
    #         print('***target',targetpath.stem)
    #         print('***to archive',new_fldr)
    #         os.replace(targetpath,archive)
    #
    #         # self.zip(archive)
    #
    #     except:
    #         print(f'error timestamping image fldr, make sure to clean {targetpath.stem} up before running agin :)')
    #
    # def zip(self, dir:str):
    #     try:  # base_name, format[, root_dir[, base_dir
    #         shutil.make_archive(archive, 'zip', self.plot_directory)
    #         # os.rmdir(archive, *, dir_fd=None)
    #     except:
    #         print(f'error zipping images, make sure to clean {targetpath.stem} up before running agin :)')

