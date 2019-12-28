#import cv2



class classification_data:
  def __init__(self):
    self.classes = {}
    self.allowed_datatypes = {
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
    return

  def get_classes(self, specific=None):
    if not specific: return list(self.classes)
    if specific == 'names': return list(self.classes.keys())
    if specific == 'datatypes': return list(self.classes.values())
    return None

  def remove_class(self, c):
    try:
      del self.classes[c]
      print('Class "{name}" removed from data'.format(name=name))
    except: return True

  def add_class(self, name, datatype, folder=None, ):
    ''' Add class to classify
    
        Arguments:
        name (str): The class name.
        datatype (str): ['image', 'video', 'audio', 'timeseries', 'text']
        folder ():
    
        Returns:
        (bool)
    
        Example:
                
    '''
    if name in self.classes.keys():
      raise Exception('Class "{name}" already present in data!'.format(name=name))
    if not datatype in self.allowed_datatypes.keys():
      raise Exception('Datatype "{cat}" not allowed!'.format(cat=datatype))
    self.classes[name] = {}
    self.classes[name]['datatype'] = datatype

  def add_to_class(self, add, toclass, being='folder', fileformats=None ):
    '''

      add (list)
      being (str): ['folder', 'files', 'arrays']
      toclass (str):
      fileformats (list/str): list of file formats, or 

      Example:
        add_to_class(add='images/moto', being='folder', toclass='moto', fileformats=['jpg', 'png'])
    '''
    if not toclass in self.classes.keys():
      raise Exception('No class "{c}" found in data!'.format(c=toclass))
    if being == 'folder':
      if not os.path.isdir(add):
        raise Exception('No foldes "{c}" found'.format(c=add))
      
      toremove = set()
      flist_orig = set(os.listdir(add))
      for f in flist_orig:
        fname, fext = os.path.splitext(f)
        if fext not in fileformats:
          toremove.add(f)
      flist = flist_orig - toremove

    return
