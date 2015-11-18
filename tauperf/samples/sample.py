from rootpy.tree import Cut
from rootpy.plotting import Hist
from rootpy import asrootpy
import ROOT


# local imports
from .db import get_file, cleanup
from .. import NTUPLE_PATH, DEFAULT_STUDENT, DEFAULT_TREE
from .. import log; log = log[__name__]
from ..parallel import FuncWorker, run_pool

class Sample(object):

    def __init__(self, cuts=None,
                 ntuple_path=NTUPLE_PATH,
                 student=DEFAULT_STUDENT,
                 tree_name=DEFAULT_TREE,
                 name='Sample',
                 label='Sample',
                 trigger=False,
                 weight_field=None,
                 **hist_decor):

        if cuts is None:
            self._cuts = Cut()
        else:
            self._cuts = cuts

        self.ntuple_path = ntuple_path
        self.student = student
        self.tree_name = tree_name
        self.name = name
        self.label = label
        self.trigger = trigger
        self.weight_field = weight_field
        log.debug(weight_field)
        self.hist_decor = hist_decor

        if 'fillstyle' not in hist_decor:
            self.hist_decor['fillstyle'] = 'solid'

    def decorate(self, name=None, label=None, **hist_decor):
        if name is not None:
            self.name = name
        if label is not None:
            self.label = label
        if hist_decor:
            self.hist_decor.update(hist_decor)
        return self

    def events(self, category=None, cuts=None, weighted=False, force_reopen=False):
        selection = Cut(self._cuts)
        if category is not None:
            selection &= self.cuts(category)
        if cuts is not None:
            selection &= cuts
        if weighted and self.weight_field is not None:
            if isinstance(self.weight_field, (list, tuple)):
                for w in self.weight_field:
                    selection *= w
            else:
                selection *= self.weight_field
        return self.draw_helper(
            Hist(1, 0.5, 1.5), '1', 
            selection, force_reopen=force_reopen)

    def cuts(self, category=None, **kwargs):
        cuts = Cut(self._cuts)
        if category is not None:
            cuts &= category.get_cuts(**kwargs)
        if self.trigger:
            cuts &= Cut('hlt_matched_to_offline == 1')
        return cuts


    def get_field_hist(self, vars, prefix=None, dummy_range=False):
        """
        """
        field_hist = {}
        for field, var_info in vars.items():
            nbins, xmin, xmax = var_info['bins'], var_info['range'][0], var_info['range'][1]
            if dummy_range:
                nbins, xmin, xmax = 1000, -2000 , 2000
            exprs = []
            if 'prefix' in var_info:
                if not isinstance(var_info['prefix'], (list)):
                    var_info['prefix'] = [var_info['prefix']]
                from .jet import Jet
                if isinstance(self, Jet) and 'true' in  var_info['prefix']:
                     var_info['prefix'].remove('true')
                if prefix is None:
                    for p in var_info['prefix']:
                        var_name = p + '_' + var_info['name']
                        exprs.append(var_name)
                else:
                    var_name = prefix + '_' + var_info['name']
                    exprs.append(var_name)
            else:
                exprs.append(var_info['name'])
            if 'scale' in var_info:
                for i, expr in enumerate(exprs):
                    exprs[i] = '{0}*{1}'.format(expr, var_info['scale'])
            log.debug(exprs)
            for expr in exprs:
                field_hist[expr] = Hist(
                    nbins, xmin, xmax, type='D')
                
        return field_hist

    def total_events(self, weighted=False, force_reopen=False):
        rfile = get_file(self.ntuple_path, self.student, force_reopen=force_reopen)
        if weighted:
            h = rfile['Nweights']
        else:
            h = rfile['Nevents']
        return h[1].value


    def records(self, **kwargs):
        ""
        ""
        from root_numpy import tree2rec
        rfile = get_file(self.ntuple_path, self.student)
        tree = rfile[self.tree_name]
        log.info('Converting tree to record array, sorry if this is long ...')
        rec = tree2rec(tree, **kwargs)
        return rec

    def draw_helper(self, hist_template, expr, selection, force_reopen=False): 
        """
        """
        rfile = get_file(self.ntuple_path, self.student, force_reopen=force_reopen)
        tree = rfile[self.tree_name]
        # use TTree Draw for now (limited to Nbins, Xmin, Xmax)
        binning = (
            hist_template.nbins(), 
            list(hist_template.xedges())[0], 
            list(hist_template.xedges())[-1])
        # ROOT.gDirectory.cd()
        hist = hist_template.Clone()
        hist.Sumw2()
        # root_string = '{0}>>{1}{2}'.format(
        #     expr, hist.name, binning)
        root_string = expr
        log.debug("Plotting {0} using selection: {1}".format(
                root_string, selection))
        log.debug('{0}: Draw {1} with \n selection: {2} ...'.format(self.name, root_string, selection))
        hist = tree.Draw(
            root_string, 
            selection=selection,
            hist=hist,
            **self.hist_decor)
        hist.title = self.label
        return hist
        # try:
        #     # hist = asrootpy(ROOT.gPad.GetPrimitive(hist.GetName()))
        #     hist = asrootpy(ROOT.gDirectory.Get(hist.GetName()))
        #     return Hist(hist, title=self.label, **self.hist_decor)
        # except:
        #     log.warning('{0}: unable to retrieve histogram for {1} with selection {2}'.format(
        #             self.name, expr, selection))
        #     return Hist(binning[0], binning[1], binning[2], title=self.label, **self.hist_decor)

    def get_hist_array(
        self, field_hist_template, 
        category=None, cuts=None, multi_proc=False):
        """
        """
        sel = self.cuts(category)
        if not cuts is None:
            sel &= cuts
        if self.weight_field is not None:
            sel *= self.weight_field
        field_hists = {}

        from .jet import JZ
        if isinstance(self, JZ):
            multi_proc = False

        if multi_proc:
            keys = [key for key in field_hist_template.keys()]
            workers = [FuncWorker(
                        self.draw_helper, 
                        field_hist_template[key], key, sel) for key in keys]
            run_pool(workers, n_jobs=-1)
            for key, w in zip(keys, workers):
                field_hists[key] = asrootpy(w.output)
        else:
            for key, hist in field_hist_template.items():
                field_hists[key] = self.draw_helper(hist, key, sel)
        return field_hists

