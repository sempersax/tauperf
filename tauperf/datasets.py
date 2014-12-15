import os
import logging
import copy
import yaml
import re

from . import NTUPLE_PATH
from . import log; log = log[__name__]

log = logging.getLogger(os.path.basename(__file__))

PATTERN = re.compile(
    '^(?P<prefix>group.phys-higgs|user.qbuat)'
    '\.(?P<type>mc\d+_\d+TeV)'
    '\.(?P<id>\d+)'
    '\.(?P<gen>Pythia8|PowhegPythia8|PowPy8)_'
    '(?P<pdf>\w+)_'
    '(?P<sample>jetjet_JZ[0-9]W|Ztautau|perf_JF17)'
    '(_(?P<dyrange>\d+M\d+|\d+M))?'
    '\.merge.AOD.(?P<tag>e\d+_s\d+_s\d+_r\d+_r\d+)'
    '\.(?P<skim>tau_trigger|higgs)'
    '\.v(?P<version>\d+)_'
    '(?P<suffix>\S+)$')



def create_samples():
    log.info('Building samples using regular expressions ...')
    SAMPLES_RAW = {}
    KEYS = []
    

    for d in os.listdir(NTUPLE_PATH):
        abs_dir = os.path.join(NTUPLE_PATH, d)
        if os.path.isdir(abs_dir):
            match = re.match(PATTERN, d)
            if match:
                info_sample = {}
                for key in match.groupdict():
                    info_sample[key] = match.group(key)
                    info_sample['dirs'] = d
                sample_key = match.group('sample')
                if match.group('dyrange'):
                    sample_key = match.group('sample') + '_' + match.group('dyrange')

                if sample_key in SAMPLES_RAW.keys():
                    SAMPLES_RAW[sample_key].append(info_sample)
                else:
                    SAMPLES_RAW[sample_key] = [info_sample]

    log.info("= + =" * 10)
    SAMPLES = {}
    for key, samples in SAMPLES_RAW.items():
        if len(samples) == 0:
            SAMPLES[key] = {}
        else:
            SAMPLES[key] = copy.deepcopy(samples[0])
            for tag in SAMPLES[key]:
                SAMPLES[key][tag] = [SAMPLES[key][tag]]
            for s in samples:
                for tag, info in s.items():
                    if not info in SAMPLES[key][tag]:
                        SAMPLES[key][tag].append(info)

    for key, sample in SAMPLES.items():
        for tag in sample:
            if len(sample[tag]) == 1:
                sample[tag] = sample[tag][0]

    for key, sample in SAMPLES.items():
        if sample.keys():
            print '=' * 50
            for k in sample.keys():
                print '\t %s  = %s' % (k, sample[k])

    return SAMPLES

def create_database(db_name='datasets.yml'):
    SAMPLES = create_samples()
    with open(os.path.join(os.path.dirname(__file__), db_name), 'w') as fdb:
        yaml.dump(SAMPLES, fdb)

def read_database(db_name='datasets.yml'):
    if os.path.exists(os.path.join(
            os.path.dirname(__file__), db_name)):
        log.info('Load %s ...' % db_name)
        with open(os.path.join(os.path.dirname(__file__), db_name)) as fdb:
            return yaml.load(fdb)
    else:
        raise Exception('The database %s does not exists' % db_name)
