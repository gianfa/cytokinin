
## ERRORS ##

def error_wrong_argument(msg):
    base = 'WrongArgument:'
    raise Exception(f'{base} {msg}')

def error_missing_argument(msg):
    base = 'MissingArgument:'
    raise Exception(f'{base} {msg}')

def error_missing_module(mod):
    ref = None
    red = ''
    if mod == 'fastai':
        ref = 'https://pypi.org/project/fastai/'
    elif mod == 'tensorflow':
        ref = 'https://pypi.org/project/tensorflow/'
    else: raise Exception('Missing argument')
    if ref: red = 'Visit {ref}'.format(ref = ref)
    msg  = 'In order to use this function you need to install \
    "{n}" module.'.format(n = mod)
    raise Exception(' '.join([msg, red]))

ERROR = {
    'missing_module': error_missing_module,
    'wrong_argument': error_wrong_argument,
    'missing_argument': error_missing_argument,
}
