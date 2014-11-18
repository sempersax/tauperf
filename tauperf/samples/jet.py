from .sample import Sample
from .. import NTUPLE_PATH
from .. import log; log = log[__name__]

class Jet(Sample):
    pass


class JZ(Jet):
    def __init__(self, cuts=None, ntuple_path=NTUPLE_PATH, **kwargs):
        super(JZ, self).__init__(cuts=cuts, ntuple_path=ntuple_path, **kwargs)
        self.sub_samples = [
            Jet(
                cuts=self._cuts, student='jetjet_JZ0', 
                name='JZ0', label='JZ0'),
            Jet(
                cuts=self._cuts, student='jetjet_JZ1', 
                name='JZ1', label='JZ1'),
            Jet(
                cuts=self._cuts, student='jetjet_JZ2', 
                name='JZ2', label='JZ2'),
            Jet(
                cuts=self._cuts, student='jetjet_JZ7W', 
                name='JZ7', label='JZ7'),
            
            ]
        self._scales = []

    def set_scales(self, scales):
        """
        """
        if isinstance(scales, (float, int)):
            for i in xrange(self.sub_samples):
                self._scales.append(scales)
        else:
            if len(scales) != len(self.sub_samples):
                log.error('Passed list should be of size {0}'.format(len(self.sub_samples)))
                raise RuntimeError('Wrong lenght !')
            else:
                for scale in scales:
                    self._scales.append(scale)
        
        log.info('Set samples scales: {0}'.format(self._scales))

    def draw_helper(self, *args):
        hist_array = []
        individual_components = False
        for s in self.sub_samples:
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
