import os
import time
import pandas as pd
import cv2

from cytokine.data import Data


def create_dataset(of, named=None):
    return Dataset().create_dataset(of, named)

class Dataset:
    def __init__(self):
        self.datas = {}
        self.datatype = None
        self.name = 'dataset'+ str(time.time()).replace('.','')
        self.allowed_dataset_types = [
            'IMAGES', 'NUMBERS', 'TEXT', 'TIMESERIES'
        ]

    def create_dataset(self, of, named=None):
        if self.datatype:
            raise Exception('This Dataset is already taken! fill it with more Data objects or build another one!')
        if not datatype in self.__allowed_datatypes.keys():
            raise Exception('Datatype "{cat}" not allowed!'.format(cat=datatype))
        self.datatype = datatype
        return self
        
        return self

    def named(self, name):
        self.name = str(name)
        return self

    def add_data(self, do):
        if type(do) != type(Data()):
            raise Exception('The "do" argument was not a Data type!')
        if do.name in self.datas.keys():
            raise Exception('This Dataset already contains a "{nm}" Data object'.format(nm=do.name))
        self.datas[do.name] = do



    # def from_folder(self, folderpath, exts=None, notallowedfiles='ignore',  notfiles='ignore', uniques=True, verbose=False):
    #     return



