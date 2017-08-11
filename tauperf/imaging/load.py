import os
import numpy as np
import tables
from tabulate import tabulate
from . import log; log = log[__name__]


log.info('loading data...')
data_dir = os.path.join(
    os.getenv('DATA_AREA'), 'tauid_ntuples', 'v12/test')
                        

def get_train_test_val(filename, title):
    log.info('Retrieving data from {0}'.format(filename))
    h5file = tables.open_file(filename, mode='r', title=title)
    train = h5file.root.data.train
    test  = h5file.root.data.test
    val   = h5file.root.data.val
    return train, test, val


def load_data(filenames, labels, equal_size=False, debug=False):

    if len(filenames) != len(labels):
        raise ValueError('filenames and labels must have the same length')

    trains = []
    tests  = []
    vals   = []
    
    for filename, label in zip(filenames, labels):
        train, test, val = get_train_test_val(filename, label)
        trains.append(train)
        tests.append(test)
        vals.append(val)
        
    if equal_size:
        log.info('Train and validate with equal size for each mode')
        size_train = min([len(train) for train in trains])
        size_val = min([len(val) for val in vals])

    if debug:
        log.info('Train with very small stat for debugging')
        size_train = min([len(train) for train in trains] + [1000])
        size_val = min([len(val) for val in vals] + [1000])
        
    if equal_size or debug:
        trains = [train[0:size_train] for train in trains]
        vals = [val[0:size_val] for val in vals]
    else:
        trains = [train.read() for train in trains]
        vals   = [val.read() for val in vals]

    tests = [test.read() for test in tests]

    headers = ["Sample", "Training", "Validation", "Testing"]
    sample_size_table = []
    for l, tr, v, te in zip(labels, trains, vals, tests):
        sample_size_table.append([l, len(tr), len(v), len(te)])
    print sample_size_table
    log.info('')
    print tabulate(sample_size_table, headers=headers, tablefmt='simple')
    log.info('')

    train_conc = np.concatenate([train for train in trains])
    test_conc = np.concatenate([test for test in tests])
    val_conc = np.concatenate([val for val in vals])

    y_train = np.concatenate([train['truthmode'] for train in trains])
    y_test = np.concatenate([test['truthmode'] for test in tests])
    y_val = np.concatenate([val['truthmode'] for val in vals])

    return train_conc, test_conc, val_conc, y_train, y_test, y_val
