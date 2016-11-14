# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 15:33:07 2016

@author: Daniel
"""

from configparser import ConfigParser
from tkinter import *
from tkinter.filedialog import askdirectory

def load_lung_sound_path():
    config = ConfigParser()
    config.read('config/config.ini')
    if not 'lung_sounds' in config:        
        foldername = askdirectory(title='Specify lung sound folder')
        config['lung_sounds'] = {'path':foldername}        
        with open('config/config.ini', 'w') as configfile:
            config.write(configfile)
        return foldername
    else:
        return config['lung_sounds']['path']
        
        
if __name__ == '__main__':
    path = load_lung_sound_path()