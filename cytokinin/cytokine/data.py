import os
import pathlib
import time
import copy
import shutil

import numpy as np
import pandas as pd
import cv2
from cytokine.config import ERROR

def take_data(datatype):
    return Data().take_data(datatype)

class Data:
    def __init__(self):
        self.name = 'data'+ str(time.time()).replace('.','')
        self.filesnames = pd.Series([], name=self.name)
        self.datatype  = None
        self.__allowed_datatypes = {
            'images': [
                #cv2 supported, ref: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
                '.bmp', '.dib', # Windows bitmaps - *.bmp, *.dib (always supported)
                '.jpeg', '.jpg', '.jpe', # JPEG files - *.jpeg, *.jpg, *.jpe
                '.jp2', # JPEG 2000 files - *.jp2
                '.png', # Portable Network Graphics - *.png
                '.webp', # WebP - *.webp
                '.pbm', '.pgm', '.ppm' '.pxm', '.pnm', # Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm 
                '.sr', '.ras', # Sun rasters - *.sr, *.ras 
                '.tiff', '.tif', # TIFF files - *.tiff, *.tif
                '.exr', # OpenEXR Image files - *.exr 
                '.hdr', '.pic',# Radiance HDR - *.hdr, *.pic (always supported)
            ],
        }
    
    def __str__(self):
        args = (
            type(self),
            self.datatype,
            len(self.filesnames),
        )
        txt = 'Object %s, of data type "%s", having %s file paths records.' % args
        return txt

    def copy(self):
        return copy.deepcopy(self)

    def length(self):
        return len(self.filesnames)

    def take_data(self, datatype):
        if self.datatype:
            raise Exception('This Data is already taken! fill it with more files or build another one!')
        if not datatype in self.__allowed_datatypes.keys():
            raise Exception('Datatype "{cat}" not allowed!'.format(cat=datatype))
        self.datatype = datatype
        return self
    
    ### UTILS ###
    def filesnames_to_ml_df(self, xcol=None, ycol=None, withAbsPath=False, labels2cat=False):
        ''' Sticks a target column to self.filesnames

        '''
        x_col = self.name
        if xcol and type(xcol) == str: x_col = xcol
        df = self.filesnames.to_frame().rename(columns={self.name: x_col})
        if not ycol:
            df['y'] = self.name
        if type(ycol) == list or type(ycol) == np.ndarray:
            if len(ycol) != len(self.filesnames):
                raise Exception('WrongArgument: ycol length is different from self.filesnames length')
            if labels2cat:
                try:
                    from keras.utils import to_categorical
                    df['y'] = np.array(ycol)
                    df['y'] = to_categorical(df['y'].values)
                except ModuleNotFoundError as e:
                    ERROR['missing_module']('keras')
            else:
                df['y'] = np.array(ycol)
        if withAbsPath:
            to_abs_path = lambda x: os.path.abspath(x) if type(x) != pathlib.PosixPath else x.absolute()
            df[x_col] = df[x_col].map(to_abs_path)
        return df

    ### STORE ###
    def store_filesnames_from_df(self, df, col='fnames', exts=None, notallowedfiles='ignore',  notfiles='ignore', uniques=True,  verbose=False):
        if not isinstance(df, type(pd.DataFrame()) ):
            raise Exception('The df argument wasn\'t a pandas DataFrame!')
        if not col in list(df.columns):
            raise Exception('Dataframe column "{col}" not found'.format(col=col))
        flist = df[col].to_list()
        return self.store_filesnames_from_list(flist, exts=exts, notallowedfiles=notallowedfiles,  notfiles=notfiles, uniques=uniques, verbose=verbose)
    
    def store_filesnames_from_list(self, flist, exts=None, notallowedfiles='ignore',  notfiles='ignore', uniques=True, verbose=False):
        ''' Store filesnames appending the good ones to the internal Series
        
            Arguments:
                flist (list): files paths list
                exts (list): list of extensions to consider
                notallowedfiles (str): whether ignore or raise exceptions on giles with different extension than exts ['ignore', 'exts']
                notfiles (str): as *notallowedfiles*, but for filepaths not recognized as files.
                uniques (str): if True drops file paths already in self.filesnames.
            Returns:
                None
            Example:
                IMGS = './cytokine/tests/mocks/imgs/'
                flist = []
                for root, dirs, files in os.walk(IMGS, topdown=False):
                    for f in files:
                        flist.append(os.path.join(root, f))
                d = take_data('images)
                d.store_filensames(flist)
        '''
        EXTS = None
        if exts:
            EXTS = self.__allowed_datatypes[self.datatype].intersection(set(exts))
            if len(EXTS) == 0:
                raise Exception('Invalid or not allowed extensions for "{tp}" files.'.format(tp=self.datatype))
        else: EXTS = self.__allowed_datatypes
        
        temp_flist=set()
        for f in flist:
            verified = True
            # verify extensions
            if not f.rpartition('.')[-1].lower() in EXTS:
                if notallowedfiles == 'except':
                    raise Exception('Other file extensions were found.')
                if notallowedfiles == 'ignore':
                    pass
            # verify existence
            if not os.path.isfile(f):
                import matplotlib.pyplot as plt
                if notfiles == 'except':
                    raise Exception('File "{fp}" not found!'.format(fp=f))
                elif notfiles == 'ignore':
                    verified = False
                else: raise Exception('Not valid value "{v}" was passed as notfiles argument'.format(v=notfiles))
            # store filename
            if verified: temp_flist.add(f)
        if uniques:
            temp_flist = list( temp_flist - set(self.filesnames.values) )
        self.filesnames = self.filesnames.append( pd.Series(list(temp_flist), name=self.name) )
        if verbose: print('{n} files added to {c} set'.format(n=len(self.filesnames), c=self.name))
        return self

    def store_filesnames_from_folder(self, folderpath, exts=None, notallowedfiles='ignore',  notfiles='ignore', uniques=True, verbose=False):
        if not os.path.isdir(folderpath):
            raise Exception('{fn} not found as folder path'.format(fn=folderpath))
        flist = []
        for root, dirs, files in os.walk(folderpath, topdown=False):
            for f in files:
                flist.append(os.path.join(root, f))
        return self.store_filesnames_from_list(flist)

    ## TO ##
    def to(self, outtype='list', array_mode=None, limit_from=0, limit_to=None ):
        map2cv2 = {
            'rgb': cv2.IMREAD_COLOR,
            'gray': cv2.IMREAD_GRAYSCALE,
            'grey': cv2.IMREAD_GRAYSCALE,
        }
        # cut the set
        if limit_to: out = self.filesnames[limit_from:limit_to]
        else: out = self.filesnames[limit_from:]
        # return the set
        if outtype == 'list': return self.filesnames.to_list()
        if outtype == 'dataframe': return self.filesnames.to_frame()
        if outtype == 'series': return self.filesnames
        if outtype == 'arrays':
            al = self.filesnames.to_list()
            if not array_mode: return [ cv2.imread(f) for f in al ]
            if not array_mode in map2cv2.keys():
                raise Exception('Invalid "array_mode" argument! it must be "rgb" or "gray"')
            return [ cv2.imread(f, map2cv2[array_mode]) for f in al ]
        raise Exception('Invalid "format" argument!')

    ## ADD DATA ##
    def add_from_data(self, do, exts=None, notallowedfiles='ignore',  notfiles='ignore', uniques=True, verbose=False):
        if type(do) != type(Data()):
            raise Exception('The "do" argument was not a Data type!')
        if do.datatype != self.datatype:
            raise Exception('Datatype of "{nm}" Data object is {dt}, different from this Data datatype'.format(nm=do.name, dt=do.datatype))
        flist = do.to('list')
        self.store_filesnames_from_list(
            flist, exts=exts, notallowedfiles=notallowedfiles,
            notfiles=notfiles, uniques=uniques, verbose=verbose
        )

    ## EXPELL ##
    def expell_to(self, path, inthisnewfolder=None, namefilesas=None, asprefix=True, ascopy=True):
        '''
            Arguments:
                path (str):
                inthisnewfolder (str): will create a folder named as this variable, where files will be put.
                namefilesas (str):
                    * None: the files hold their names
                    * 'data': the files will renamed as their self.name
                    * 'folder: the files
                asprefix (bool):
                    * True: the files will be saved named as ASPREFIX_ORIGINALNAME.extension.
                    * False: the files will hold their original name.
                ascopy (bool): Whether to save files as copy or move them from their original path.
            
            Returns:
                (str): path to the folder having the expelled files
            
            Example
        '''
        if inthisnewfolder: assert type(inthisnewfolder) == str
        # TODO: if path and if path is dir
        # TODO: asserts....
        while path[-1] == '/': path = path[:-1]
        folder = None
        if inthisnewfolder:
            folder =  os.path.join(path, inthisnewfolder)
            os.mkdir(folder)
            folder_name = inthisnewfolder
        else:
            folder = path
            folder_name = os.path.split(path)[-1]
        for i, fpath in enumerate(self.filesnames): #TODO: riferimento al file
            # file name
            fname = os.path.split(fpath)[-1]
            if namefilesas == 'data' and asprefix: fname = '_'.join([self.name, fname])
            elif namefilesas == 'data' and not asprefix: fname = '_'.join([self.name, '_', i])
            if namefilesas == 'folder' and asprefix: fname = '_'.join([folder_name, fname])
            elif namefilesas == 'folder' and not asprefix: fname = '_'.join([folder_name, '_', i])
            # execution
            dst = os.path.join(folder, fname)
            if ascopy: shutil.copy(fpath, dst)
            else: shutil.move(fpath, dst)
        return folder

    # TODO
    def expell_compressed_to(self, path, asformat='tar',**kwargs):
        pass

    
    ## EXPORT ##
    def export_to_keras(self, imagedatagenerator_args={}, flowfromdf_args={}, labels2cat=True):
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator 
        except ModuleNotFoundError as e:
            ERROR['missing_module']('tensorflow')
        # https://colab.research.google.com/drive/1rH6ajx5MYKAH72EfDhWGaLvCCSzOBegf
        
        dg = ImageDataGenerator(**imagedatagenerator_args)
        x_col = 'filesname'
        y_col = 'y'
        ffdf = dg.flow_from_dataframe(
            dataframe = self.filesnames_to_ml_df(xcol=x_col),
            x_col = x_col,
            y_col = y_col,
            labels2cat = labels2cat,
            **flowfromdf_args
        )
        return ffdf

    def export_to_fastAI(self, imagedatabunch_args={}):
        # https://docs.fast.ai/vision.data.html#ImageDataBunch.from_df
        # from fastai.vision import ImageDataBunch
        # data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
        # return data
        try:
            from fastai.vision import ImageDataBunch, ImageList
            from fastai.vision.transform import get_transforms
        except ModuleNotFoundError as e:
            ERROR['missing_module']('fastai')
        df = self.filesnames_to_ml_df(withAbsPath=False)
        fastai_idb  = ImageDataBunch.from_df(path='', df=df, **imagedatabunch_args)
        return fastai_idb
    
    # TODO
    def export_to_pytorch(self):
        # https://discuss.pytorch.org/t/dataloading-using-pandas/33833/2
        # df = pd.read_csv(csvFilePath)
        # tmp = df.values
        # result = torch.from_numpy(tmp)
        pass
    
    # TODO
    def export_to_tensorflow(self):
        # https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
        # dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
        pass