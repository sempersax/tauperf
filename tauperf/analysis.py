import re

from . import log; log = log[__name__]
from . import samples
from .categories import CATEGORIES
from . import NTUPLE_PATH
VAR_PATTERN = re.compile('((?P<prefix>hlt|off|true)_)?(?P<var>[A-Za-z0-9_]+)(\*(?P<scale>\d+\.\d*))?$')


class Analysis(object):
    
    def __init__(self, 
                 ntuple_path=NTUPLE_PATH,
                 use_drellyan=False,
                 use_jz_slices=False):
        log.info('Analysis object is being instantiated')
        if use_drellyan:
            log.info('Use Drell-Yan simulation')
            self.tau = samples.DY(
            ntuple_path=ntuple_path,
            # weight_field='mc_event_weight',
            name='tau', label='Real #tau_{had}',
            color='#00A3FF')
        else:
            self.tau = samples.Tau(
                ntuple_path=ntuple_path,
                name='tau', label='Real #tau_{had}',
                color='#00A3FF')

        if use_jz_slices:
            self.jet = samples.JZ(
                ntuple_path=ntuple_path,
                name='jet', 
                label='Fake #tau_{had}',
                weight_field='mc_event_weight', 
                color='#00FF00')
        else:
            self.jet = samples.Jet(
                ntuple_path=ntuple_path,
                student='jetjet_JZ1W',
                name='jet', 
                label='Fake #tau_{had}',
                color='#00FF00')
            
        log.info('Analysis object is instantiated')

    def iter_categories(self, *definitions, **kwargs):
        names = kwargs.pop('names', None)
        for definition in definitions:
            for category in CATEGORIES[definition]:
                if names is not None and category.name not in names:
                    continue
                log.info("")
                log.info("=" * 40)
                log.info("%s category" % category.name)
                log.info("=" * 40)
                log.info("Cuts: %s" % self.tau.cuts(category))
                yield category

    def get_hist_samples_array(self, vars, prefix, **kwargs):
        """
        """
        field_hist_tau = self.tau.get_field_hist(vars, prefix)
        log.debug('Retrieve Tau histograms')
        field_hist_tau = self.tau.get_hist_array(field_hist_tau, **kwargs)
        field_hist_jet = self.jet.get_field_hist(vars, prefix)
        log.debug('Retrieve Jet histograms')
        field_hist_jet = self.jet.get_hist_array(field_hist_jet, **kwargs)
        hist_samples_array = {}
        for key in field_hist_tau:
            match = re.match(VAR_PATTERN, key)
            if match:
                hist_samples_array[match.group('var')] = {
                    'tau': field_hist_tau[key],
                    'jet': field_hist_jet[key]
                }
            else:
                log.warning('No pattern matching for {0}'.format(key))
        return hist_samples_array

    def get_hist_signal_array(self, vars, prefix1, prefix2, **kwargs):
        """
        """
        field_hist_tau_1 = self.tau.get_field_hist(vars, prefix1)
        field_hist_tau_2 = self.tau.get_field_hist(vars, prefix2)
        log.debug('Retrieve Tau histograms')
        field_hist_tau_1 = self.tau.get_hist_array(field_hist_tau_1, **kwargs)
        field_hist_tau_2 = self.tau.get_hist_array(field_hist_tau_2, **kwargs)
        
        hist_samples_array = {}
        for key in field_hist_tau_1:
            match = re.match(VAR_PATTERN, key)
            if match:
                field_hist_tau_1[key].title += ' ({0})'.format(prefix1)
                hist_samples_array[match.group('var')] = {prefix1: field_hist_tau_1[key]}
        for key in field_hist_tau_2:
            match = re.match(VAR_PATTERN, key)
            if match:
                field_hist_tau_2[key].title += ' ({0})'.format(prefix2)
                hist_samples_array[match.group('var')][prefix2] = field_hist_tau_2[key]
        return hist_samples_array
