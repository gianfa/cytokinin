from tkinter import Tk
from tkinter import filedialog
from tkinter.messagebox import showerror, showwarning, showinfo

from pathlib import Path
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

def select_folder():
    '''
        Refs:
            https://stackoverflow.com/questions/20790926/ipython-notebook-open-select-file-with-gui-qt-dialog
    '''
    Tk().withdraw() # keep the root window from appearing
    folder = filedialog.askdirectory()
    # if ensure_is_notempty:
    #     while len(os.listdir(folder)) == 0:
    #         msg = f'The folder {folder} appears to be empty! Please another not empty folder'
    #         showerror('Empty folder Error', msg)
    #         folder = filedialog.askdirectory()
    # log.debug(f'loaded imgs from {folder}')
    return Path(folder)


def select_filename():
    '''
    '''
    Tk().withdraw() # keep the root window from appearing
    filename = filedialog.askopenfilename()
    return Path(filename)