'''
Run from the parent folder
    cytokinin/../$ pytest -vv
'''

import os
import pytest
import copy
import shutil
import pandas as pd
import numpy as np
from cytokinin.data import *

# pytest -v -s

IMGS = './cytokinin/tests/mocks/imgs/'

def get_mocked_imgs_list(fold):
    print(os.getcwd())
    print(IMGS)
    return [IMGS + fold + '/' + x for x in os.listdir(os.path.join(IMGS, fold))]



def test_add_from_data():
    flist = get_mocked_imgs_list('dog')
    dwell = take_data('images').store_filesnames_from_list(flist[:10])
    a2 = take_data('images').store_filesnames_from_list(flist[5:20])
    dwell.add_from_data(a2)
    assert len(dwell.filesnames) == 20

def test_to():
    flist = get_mocked_imgs_list('dog')[:10]    
    d = take_data('images').store_filesnames_from_list(flist)
    assert type(d.to('list')) == list
    assert type(d.to('dataframe')) == type(pd.DataFrame())
    assert type(d.to('series')) ==  type(pd.Series())
    array_list = d.to('arrays')
    assert type(array_list) ==  list and type(array_list[0]) == type(np.array([]))
    return

def test_store_filesnames_from_df():
    flist = get_mocked_imgs_list('dog')
    df = pd.DataFrame({'x': flist})
    d = take_data('images')
    # correct
    d.store_filesnames_from_df(df, 'x')
    assert len(d.filesnames) > 0
    # except if...
    with pytest.raises(Exception):
        # not a dataframe
        df2 = list('a', 'b')
        assert d.store_filesnames_from_df(df2)
        # wrong column name
        assert d.store_filesnames_from_df(df)
        assert d.store_filesnames_from_df(df, col='GABIBBO')


def test_store_filesnames_from_list():
    flist = get_mocked_imgs_list('dog')
    d = take_data('images')
    # verify extensions #
    flist2 = flist.copy()
    flist2[7] = flist[7][:-3] + 'txt'
    flist2[8] = flist[8][:-3] + 'strangeFormat'
    # raise Exception if..
    with pytest.raises(Exception):
        # if a not allowed extension has been provided -> except
        assert d.store_filesnames_from_list(flist2, exts=['uglyformat'])
        # if flists2 contains files with other extensions than allowed -> except
        assert d.store_filesnames_from_list(flist2, exts=None, notallowedfiles='except')
        # if at least one file doesn't exists -> except
        assert d.store_filesnames_from_list(flist2, notfiles='except' )
    # keep uniques
    flist3_1 = flist[:10].copy()
    flist3_2 = flist[7:16].copy()
    d2 = take_data('images').store_filesnames_from_list(flist3_1)
    d2.store_filesnames_from_list(flist3_2, uniques=True)
    assert len(d2.filesnames) == 16


def test_store_filesnames_from_folder():
    d = take_data('images')
    d2 = copy.copy(d)
    d2.store_filesnames_from_folder(os.path.join(IMGS, 'dog'))
    with pytest.raises(Exception):
        assert d2.store_filesnames_from_folder('gigio/')

def test_expell_to():
    flist = get_mocked_imgs_list('dog')[:5]
    c = take_data('images').store_filesnames_from_list(flist)
    path = './cytokinin/tests/mocks/imgs/'
    inthisnewfolder = 'new_images'
    namefilesas = None
    res = c.expell_to(path, inthisnewfolder, namefilesas)
    assert type(res) == str
    assert os.path.split(res)[-1] == inthisnewfolder
    dst_path = os.path.join(path, inthisnewfolder)
    assert len(os.listdir(dst_path)) == len(flist)
    # now clean the fresh made directory
    shutil.rmtree(res)
    assert not inthisnewfolder in os.listdir(path)


def test_export_to_keras():
    flist = get_mocked_imgs_list('dog')[:5]
    c = take_data('images').store_filesnames_from_list(flist)
    ke_gen = c.export_to_keras()
    assert str(type(ke_gen)) == "<class 'keras_preprocessing.image.dataframe_iterator.DataFrameIterator'>"

#def test_export_to_fastAI()

# transformations = transforms.Compose([
#     transforms.Resize((224,224),interpolation=Image.NEAREST), # NEEDED if you want to use *batch_size*
#     transforms.ToTensor()
# ])
# t_df = pd_Dataset(mldf, transformations, colors='L')
# t_df
# ## TEST ##
# def test_pd_Dataset_colors():
#     transformations = transforms.Compose([
#         transforms.Resize((128,128),interpolation=Image.NEAREST), # NEEDED if you want to use *batch_size*
#         transforms.ToTensor()
#     ])
#     t_df = pd_Dataset(mldf, transformations)
#     sh = [im[0].numpy().shape for im in t_df]
#     assert sum([s[0] != 1 for s in sh]) > 0, \
#     'Error: with no colors defined, a list of different channels sizes (dim 0 of tensors) is expected!'
    
#     # colors = 'L' # PIL greyscale
#     t_df = pd_Dataset(mldf, transformations, colors='L')
#     sh = [im[0].numpy().shape for im in t_df]
#     assert sum([s[0] != 1 for s in sh]) == 0, \
#     'BatchShapeIncongruent: all images in batch must have only 1 channel'
    
