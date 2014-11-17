from . import log; log = log[__name__]

from . import samples
from .categories import CATEGORIES


class Analysis(object):
    
    def __init__(self):

        self.tau = samples.Tau(
            name='tau', label='Real #tau_{had}',
            color='#00A3FF')

        self.jet = samples.Jet(
            name='jet', 
            student='jetjet_JZ7W',
            label='Fake #tau_{had}',
            color='#00FF00')


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
                log.info("Cuts: %s" % self.tau.cuts(category,))
                yield category

    def get_hist_samples_array(self, vars, prefix, category=None, cuts=None):
        """
        """
        field_hist_tau = self.tau.get_field_hist(vars, prefix)
        log.info(field_hist_tau)
        log.debug('Retrieve Tau histograms')
        field_hist_tau = self.tau.get_hist_array(field_hist_tau, category=category, cuts=cuts)
        field_hist_jet = self.jet.get_field_hist(vars, prefix)
        log.debug('Retrieve Jet histograms')
        field_hist_jet = self.jet.get_hist_array(field_hist_jet, category=category, cuts=cuts)
        hist_samples_array = {}
        for key, var_info in vars.items():
            if 'prefix' in var_info:
                var_name = prefix + '_' + var_info['name']
            else:
                var_name = var_info['name']
            hist_samples_array[key] = {
                'tau': field_hist_tau[var_name],
                'jet': field_hist_jet[var_name]
                }
        return hist_samples_array

    def get_hist_signal_array(self, vars, prefix1, prefix2, category=None, cuts=None):
        """
        """
        field_hist_tau_1 = self.tau.get_field_hist(vars, prefix1)
        field_hist_tau_2 = self.tau.get_field_hist(vars, prefix2)
        log.debug('Retrieve Tau histograms')
        field_hist_tau_1 = self.tau.get_hist_array(field_hist_tau_1, category=category, cuts=cuts)
        field_hist_tau_2 = self.tau.get_hist_array(field_hist_tau_2, category=category, cuts=cuts)

        hist_samples_array = {}
        for key, var_info in vars.items():
            var_name = var_info['name']
            if 'prefix' in var_info:
                hist_samples_array[key] = {
                    prefix1: field_hist_tau_1[prefix1+'_'+var_name],
                    prefix2: field_hist_tau_2[prefix2+'_'+var_name]
                    }
            else:
                hist_samples_array[key] = {
                    prefix1: field_hist_tau_1[var_name],
                    prefix2: field_hist_tau_2[var_name]
                    }
        return hist_samples_array
