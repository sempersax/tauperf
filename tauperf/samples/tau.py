from rootpy.tree import Cut

from .sample import Sample
from .. import NTUPLE_PATH
from .. import log; log = log[__name__]

class Tau(Sample):
    
    def __init__(self, *args, **kwargs):
        super(Tau, self).__init__(*args, **kwargs)

    def cuts(self, *args, **kwargs):
        cut = super(Tau, self).cuts(*args, **kwargs)
        cut &= Cut('true_matched_to_offline != -1')
        return cut

DYRANGES = [
    '180M250',
    '250M400',
    '400M600',
    # '600M800',
    '800M1000',
    '1000M1250',
    '1250M1500',
    '1500M1750',
    # '1750M2000',
    '2000M2250',
    '2250M2500',
    '2500M2750',
    '2750M3000',
    '3000M',
]

XSEC_FILTER = {
    '180M250': (0.0029196, 1.0000E+00),
    '250M400': (0.0010817, 1.0000E+00),
    '400M600': (0.00003739, 1.0000E+00),
    '600M800': (0.00003739, 1.0000E+00),
    '800M1000': (0.000010604, 1.0000E+00),
    '1000M1250': (0.0000042578, 1.0000E+00),
    '1250M1500': (0.0000014218, 1.0000E+00),
    '1500M1750': (0.00000054521, 1.0000E+00),
    '1750M2000': (0.0000002299, 1.0000E+00),
    '2000M2250': (0.00000010386, 1.0000E+00),
    '2250M2500': (0.000000049399, 1.0000E+00),
    '2500M2750': (.000000024454, 1.0000E+00),
    '2750M3000': (0.000000012489, 1.0000E+00),
    '3000M':     (0.00000001427, 1.0000E+00),
}

class DY(Tau):
    def __init__(self, cuts=None, ntuple_path=NTUPLE_PATH, **kwargs):
        super(Tau, self).__init__(cuts=cuts, ntuple_path=ntuple_path, **kwargs)
        self._sub_samples = [Tau(
                ntuple_path=ntuple_path,
                cuts=self._cuts, 
                student='Ztautau_{0}'.format(dy_range), 
                name='{0}'.format(dy_range), 
                label='{0}'.format(dy_range))
                for dy_range in DYRANGES]
        self._scales = []
        for s in self._sub_samples:
            log.info('{0}: events = {1}, weighted = {2}, xsec = {3}, filter = {4}'.format(
                    s.name, s.total_events(), s.total_events(weighted=True), 
                    XSEC_FILTER[s.name][0], XSEC_FILTER[s.name][1]))
            self._scales.append(XSEC_FILTER[s.name][0] * XSEC_FILTER[s.name][1] / s.total_events(weighted=True))
        log.info(self.scales)

    @property
    def components(self):
        return self._sub_samples

    @property
    def scales(self):
        return self._scales

    def set_scales(self, scales):
        """
        """
        if isinstance(scales, (float, int)):
            for i in xrange(self._sub_samples):
                self._scales.append(scales)
        else:
            if len(scales) != len(self._sub_samples):
                log.error('Passed list should be of size {0}'.format(len(self._sub_samples)))
                raise RuntimeError('Wrong lenght !')
            else:
                for scale in scales:
                    self._scales.append(scale)
        
        log.info('Set samples scales: {0}'.format(self._scales))

    def draw_helper(self, *args, **kwargs):
        hist_array = []
        individual_components = kwargs.pop('individual_components', False)
        #         individual_components = 
        for s in self._sub_samples:
            h = s.draw_helper(*args)
            hist_array.append(h)
        if individual_components:
            return hist_array
        else:
            if len(self._scales) != len(hist_array):
                log.error('The scales are not set properly')
                raise RuntimeError('scales need to be set before calling draw_helper')
            hsum = hist_array[0].Clone()
            hsum.reset()
            hsum.title = self.label
            for h, scale in zip(hist_array, self._scales):
                hsum += scale * h
            return hsum
