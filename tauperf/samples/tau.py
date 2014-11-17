from rootpy.tree import Cut
from .sample import Sample


class Tau(Sample):
    
    def __init__(self, *args, **kwargs):
        super(Tau, self).__init__(*args, **kwargs)

    def cuts(self, *args, **kwargs):
        cut = super(Tau, self).cuts(*args, **kwargs)
        cut &= Cut('true_matched_to_offline != -1')
        return cut
