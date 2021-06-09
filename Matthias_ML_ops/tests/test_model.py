import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append(r"..")


import torch



def test_model():
    return