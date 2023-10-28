import glob
import yaml
import random
import numpy as np
import os
import shutil

if __name__=='__main__':
    src=glob.glob('data/format_data/eval/0/*')
    src=src[:100]
    des='data/format_data/train/34'
    for aaa  in src:
        shutil.move(aaa,des)

    
