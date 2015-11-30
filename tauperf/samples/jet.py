from .sample import Sample
from .. import NTUPLE_PATH
from .. import log; log = log[__name__]
from rootpy.tree import Cut

class Jet(Sample):
    pass

class DataJet(Jet):

    def __init__(self, *args, **kwargs):
        super (DataJet, self).__init__(*args, **kwargs)

    def cuts(self, *args, **kwargs):
        cut = super(DataJet, self).cuts(*args, **kwargs)
        cut &= Cut('met < 100000.')
        return cut


XSEC_FILTER = {
    'JZ0': (78420000, 9.7193E-01),
    'JZ1': (78420000, 2.7903E-04),
    'JZ2': (57312, 5.2261E-03),
    'JZ3': (1447.8, 1.8068E-03),
    'JZ4': (23.093, 1.3276E-03),
    'JZ5': (0.23793, 5.0449E-03),
    'JZ6': (0.0054279, 5.4279E-03),
    'JZ7': (0.00094172, 6.7141E-02),
}

class JZ(Jet):
    def __init__(self, cuts=None, ntuple_path=NTUPLE_PATH, **kwargs):
        super(JZ, self).__init__(cuts=cuts, ntuple_path=ntuple_path, **kwargs)
        self._sub_samples = [
            # Jet(
            #     ntuple_path=ntuple_path,
            #     cuts=self._cuts, student='jetjet_JZ0', 
            #     name='JZ0', label='JZ0'),
            Jet(
                ntuple_path=ntuple_path,
                cuts=self._cuts, student='jz1w', 
                name='JZ1', label='JZ1'),
            Jet(
                ntuple_path=ntuple_path,
                cuts=self._cuts, student='jz2w', 
                name='JZ2', label='JZ2'),
            Jet(
                ntuple_path=ntuple_path,
                cuts=self._cuts, student='jz3w', 
                name='JZ3', label='JZ3'),
            Jet(
                ntuple_path=ntuple_path,
                cuts=self._cuts, student='jz4w', 
                name='JZ4', label='JZ4'),
            Jet(
                ntuple_path=ntuple_path,
                cuts=self._cuts, student='jz5w', 
                name='JZ5', label='JZ5'),
            Jet(
                ntuple_path=ntuple_path,
                cuts=self._cuts, student='jz6w', 
                name='JZ6', label='JZ6'),
            Jet(
                ntuple_path=ntuple_path,
                cuts=self._cuts, student='jz7w', 
                name='JZ7', label='JZ7'),
            ]
        self._scales = []
        for s in self._sub_samples:
            log.debug('{0}: events = {1}, weighted = {2}, xsec = {3}, filter = {4}'.format(
                    s.name, s.total_events(), s.total_events(weighted=True), 
                    XSEC_FILTER[s.name][0], XSEC_FILTER[s.name][1]))
            self._scales.append(XSEC_FILTER[s.name][0] * XSEC_FILTER[s.name][1] / s.total_events())
        log.debug('Scales: %s' % self.scales)

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
