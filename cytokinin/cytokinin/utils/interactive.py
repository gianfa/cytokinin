import tkinter as tk
from tkinter import Tk
from tkinter import filedialog
from tkinter.messagebox import showerror, showwarning, showinfo

from pathlib import Path
import logging

from .funx import infer_file_cols_dtypes

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

def select_folder(title='Select a folder'):
    '''
        Refs:
            https://stackoverflow.com/questions/20790926/ipython-notebook-open-select-file-with-gui-qt-dialog
    '''
    Tk().withdraw() # keep the root window from appearing
    folder = filedialog.askdirectory(title=title)
    # if ensure_is_notempty:
    #     while len(os.listdir(folder)) == 0:
    #         msg = f'The folder {folder} appears to be empty! Please another not empty folder'
    #         showerror('Empty folder Error', msg)
    #         folder = filedialog.askdirectory()
    # log.debug(f'loaded imgs from {folder}')
    return Path(folder)


def select_filename(title='Select a file'):
    '''
    '''
    Tk().withdraw() # keep the root window from appearing
    filename = filedialog.askopenfilename(title=title)
    return Path(filename)

def select_df_col(filepath, ftype='csv'):
    ''' Df column selector
        
        It opens a dialog listing the columns names of the data
        at *filepath*, with theis inferred data type.
        You can chose one of those columns and get its name.
        
        Args:
            filepath (str/PosixPath):
            ftype (str):
        
        Returns:
            (str)
    '''
    types = infer_file_cols_dtypes(filepath)

    def close_window(): 
        root.destroy()

    def on_closing():
        root.destroy()
    
    root = tk.Tk()
    v = tk.StringVar()
    v.set('')  # initializing the choice, i.e. Python
    tk.Label(root, 
             text="""Select labels column""",
             justify = tk.LEFT,
             padx = 20).pack()

    for col, col_type in types.items():
        tk.Radiobutton(root, 
                      text=str(f'"{col}"/{col_type}'),
                      padx = 20, 
                      variable=v, 
                      command=None,
                      font='Arial 11 bold',
                      value=col).pack(anchor=tk.W)
    tk.Button(text ="Confirm", command = close_window).pack(side="bottom")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
    return v.get()