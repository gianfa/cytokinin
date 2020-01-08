'''
    Currently the main class of the library
    TODO:
        * filesnames_to_ml_df: generate from filesnames and labels
        * fix exports in order to make stupidly easy to be used for fit()
        * fix interactive labeling for CSV
        * replace "rais Exception ..." expressions with ERROR[.....] from config.py
        * expell_to: add compress parameter
'''

import os
import pathlib
from pathlib import Path, PosixPath
import time
import copy
import shutil
import logging

import numpy as np
import pandas as pd
import cv2
from cytokinin.config import ERROR

from .utils import interactive
from .utils.funx import infer_file_cols_dtypes, ar1hot

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.info('CYTOKININ LOADED')

def take_data(datatype):
    return Data().take_data(datatype)

class Data:
    '''
        Class of a Data root. Each Data root can be of only one datatype.
    '''
    def __init__(self):
        self.name = self.new_name()
        self.parents = [] 
        self.filesnames = pd.Series([], name=self.name)
        self.labels = pd.Series([], name=self.name)
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
            len(self.labels)
        )
        txt = 'Object %s, of data type "%s"\n, \
            %s file paths stored\n \
            %s labels stored.' \
            % args
        return txt

    def __len__(self):
        '''
            Number of filesname-label pairs
        '''
        if len(self.filesnames) != len(self.labels):
            return 0
        return self.filesnames
    
    
    def new_name(self):
        pred = 'data'
        tstamp = str(time.time()).replace('.','')
        new_name = f'{pred}{tstamp}'
        return new_name

    def copy(self):
        return copy.deepcopy(self)

    def take_data(self, datatype):
        if self.datatype:
            raise Exception('This Data is already taken! fill it with more files or build another one!')
        if not datatype in self.__allowed_datatypes.keys():
            raise Exception('Datatype "{cat}" not allowed!'.format(cat=datatype))
        self.datatype = datatype
        return self
    
    ### ML dataset ###
    def filesnames_to_ml_df(self,
        xcol_name='X',ycol_name='y',ycol=None,
        withAbsPath=False, x_as='str', y_as='str'):
        ''' Sticks a target column to self.filesnames
            Args:
                xcol_name (str):
                ycol_name (str):
                ycol (list/numpy.array):
                withAbsPath (bool):
                y_as (str): y col values type
                    * 'str': strings;
                    * 'categorical': Pandas 'category' type;
                    * 'classnum': int referred to the class;
                    * '1hot': binary matrix in 1-hot encoding, referred to the class.
        '''
        xcolName = self.name
        if xcol_name and type(xcol_name) == str: xcolName = xcol_name
        df = self.filesnames.to_frame()

        # if we have a ycol -> append it as y col
        if ycol and type(ycol) == list or type(ycol) == np.ndarray:
            if len(ycol) != len(self.filesnames):
                raise Exception('WrongArgument: ycol length is different from self.filesnames length')
            df['y'] = np.array(ycol)
        elif not ycol and len(self.labels) == 0:
            df['y'] = self.name # no ycol -> assume all x are from one category
        else: df['y'] = self.labels
        df.rename(columns={self.name: xcolName, 'y': ycol_name}, inplace=True) 

        if withAbsPath:
            to_abs_path = lambda x: os.path.abspath(x) if type(x) != pathlib.PosixPath else x.absolute()
            df[xcolName] = df[xcolName].map(to_abs_path)
        if x_as == 'str':
            df[xcolName] = df[xcolName].astype(str)

        if y_as == 'str':
            df[ycol_name] = df[ycol_name].astype(str)
        if y_as == 'categorical':
            df[ycol_name] = df[ycol_name].astype("category")
        if y_as == '1hot':
            df[ycol_name] = pd.get_dummies(df[ycol_name]).values
            mx = df[ycol_name].unique().max()
            df[ycol_name] = df[ycol_name].apply(lambda x: ar1hot(x, mx))
        if y_as == 'classnum':
            df[ycol_name] = pd.get_dummies(df[ycol_name]).values 
        return df

    #TODO
    def export_dataset_as(self, dstype):
        if dstype not in ['dataframe', 'csv']:
            ERROR['wrong_argument']('')
        return  

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
                IMGS = './cytokinin/tests/mocks/imgs/'
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
        for fl in flist:
            f = Path(fl)
            verified = True
            # verify extensions
            if not f.suffix in EXTS:
                if notallowedfiles == 'except':
                    raise Exception('Other file extensions were found.')
                if notallowedfiles == 'ignore':
                    pass
            # verify existence
            if not f.is_file():
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

    def store_filesnames_from_folder(self, folderpath=None, include_subdirs=True, exts=None, notallowedfiles='ignore',  notfiles='ignore', uniques=True, verbose=False, gui=False):
        '''
            folderpath,
            include_subdirs,
            exts,
            notallowedfiles,
            notfiles,
            uniques,
            verbose,
            gui (bool): if True allows you to select af folder from a dialog 
        '''
        if not folderpath and not gui:
            ERROR['missing_argument'](f'You must specify at least one argument between \'folderfile\' or \'gui\'!')
        folder = folderpath
        if gui:
            folder = interactive.select_folder()
        if not os.path.isdir(folder):
            raise Exception('{fn} not found as folder path'.format(fn=folder))
        flist = []
        if not include_subdirs:
            flist = os.listdir(folder)
        else:
            for root, dirs, files in os.walk(folder, topdown=False):
                for f in files:
                    flist.append(os.path.join(root, f))
        return self.store_filesnames_from_list(flist)


    ### UTILS ###
    def is_df(self, o):
        return isinstance(o, pd.DataFrame)

    def is_series(self, o):
        return isinstance(o, pd.Series)

    ### LABEL ###
    def label_from_folder(self):
        '''
            Reads the path of the stored filesnames and uses
            its parent folder name as label.
        '''
        if len(self.filesnames) == 0:
            log.warn(f'{self.name} Data hasn\'t any filesname to label yet')
            return
        self.labels = self.filesnames.apply(lambda x: x.parent.name)
        return

    def label_from_csv(self, filepath=None, col=None, gui=False, **kwargs):
        '''
            User must provide a col name
            Args:
                filepath (str): filepath
                col (str):
                gui (bool):
        '''
        #TODO: maybe provide a wizard to get columns names of csv
        #
        if len(self.filesnames) == 0:
            log.warn(f'{self.name} Data hasn\'t any filesname to label yet')
            return
        if not filepath and not gui:
            ERROR['missing_argument'](f'You must specify at least one argument between \'filepath\' or \'gui\'!')
        fpath = filepath
        if not gui:
            if (type(filepath) != str and type(filepath) != PosixPath): raise Exception(f'Wrong argument: You must provide a CSV path a str or PosixPath, instead it was {type(filepath)}')
            if not col or type(col) != str:
                raise Exception('You must provide a col name as str')
        else:
            log.debug(f'This is an experimental feature, therefore may not work properly')
            fpath = interactive.select_filename(title='Select a CSV file')
            col = interactive.select_df_col(fpath, ftype='csv')
        log.debug(f'COL: {col}')
        df = pd.read_csv(fpath, usecols=[col])
        fn_len = len(self.filesnames)
        if not len(df) == fn_len:
            raise Exception(f'Mismatching dimensions between stored filenames {fn_len} and labels {len(df)}!')
        self.labels = df[col]
        return

    ## SHOW ##
    #TODO: Sliceable as MutableSequence https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping

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
        if outtype == 'list': return self.filesnames.astype(str).to_list()
        if outtype == 'pathlist': return self.filesnames.to_list()
        if outtype == 'dataframe': return self.filesnames.to_frame()
        if outtype == 'series': return self.filesnames
        if outtype == 'arrays':
            al = self.filesnames.to_list()
            if not array_mode: return [ cv2.imread(str(f)) for f in al ]
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
        return self

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
        
        xcol_name = 'filename'
        ycol_name = 'class'
        mldf = self.filesnames_to_ml_df(
                    xcol_name=xcol_name,
                    ycol_name=ycol_name,
                    x_as='str',
                    y_as='',
                )
        dg = ImageDataGenerator(**imagedatagenerator_args)
        ffdf = dg.flow_from_dataframe(
            dataframe = mldf,
            class_mode='categorical',
            **flowfromdf_args
        )
        nclx = len(mldf[ycol_name].unique())
        if nclx < 2:
            ertxt = f'The dataset contains only {nclx} classes. \
                Keep in mind that you must provide at least two classes\
                if you want to perform a fit in keras 😊.'
            log.warning(ertxt)
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