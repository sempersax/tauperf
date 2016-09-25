# ROOT/rootpy imports
from rootpy.tree import Cut
from rootpy.plotting import Hist, Hist2D
from rootpy import asrootpy
from root_numpy import fill_hist
import ROOT
import numpy as np
from numpy.lib import recfunctions

# local imports
from .db import get_file, cleanup
from .. import NTUPLE_PATH, DEFAULT_STUDENT, DEFAULT_TREE
from . import log; log = log[__name__]
from ..parallel import FuncWorker, run_pool

import cache

class Sample(object):

    def __init__(
        self, cuts=None,
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

        if weight_field is not None:
            if isinstance(weight_field, (list, tuple)):
                self.weight_field = weight_field
            elif isinstance(weight_field, str):
                self.weight_field = [weight_field]
            else:
                raise ValueError('wrong type for weight_field')
            log.info('{0}: weights are {1}'.format(self.name, self.weight_field))
        else:
            self.weight_field = None

        self.hist_decor = hist_decor
        self._branches = None

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

    def records_helper(self, **kwargs):
        ""
        ""
        from root_numpy import tree2array
        rfile = get_file(self.ntuple_path, self.student)
        tree = rfile[self.tree_name]
        log.info('{0}: converting tree to record array, sorry if this is long ...'.format(self.name))
        rec = tree2array(tree, **kwargs).view(np.recarray)
        if self.weight_field is not None:
            weights = reduce(np.multiply,
                             [rec[br] for br in self.weight_field])
            rec = recfunctions.rec_append_fields(
                rec,
                names='weight',
                data=weights,
                dtypes='f8')
        return rec

    @cache.memoize_or_nothing
    def records(self, **kwargs):
        return self.records_helper(
            branches=self.branches, **kwargs)

    @property
    def branches(self):
        return self._branches

    @branches.setter
    def branches(self, b):
        if not isinstance(b , (list, tuple, str)):
            raise ValueError
        self._branches = list(b)
        
        log.info(50 * '-')
        log.info('Sample {0}, activating the following branches:'.format(
                self.name))
        for branch in self._branches:
            log.info('\t' + branch)

        if self.weight_field is not None:
            for field in self.weight_field:
                log.info('\t' + field)
                self._branches.append(field)
    
    def array(self, **kwargs):
        ""
        ""
        from root_numpy import rec2array
        rec = self.records(**kwargs)
        arr = rec2array(rec)
        return arr
    


    def draw_helper(
        self, hist_template, 
        expr, selection, 
        force_reopen=False): 
        """
        Arguments
        ---------
        hist_template: rootpy Hist, template histogram
        expr: str expression to draw
        selection: str selection (TCut)
        """
        rec = self.records(
            selection=selection.GetTitle())

        hist = hist_template.Clone()
        hist.Sumw2()
        root_string = expr
        log.debug('{0}: Draw {1} with \n selection: {2} ...'.format(
                self.name, root_string, selection))

        if self.weight_field is not None:
            fill_hist(hist, rec[expr], rec['weight'])
        else:
            fill_hist(hist, rec[expr])

        hist.title = self.label
        return hist

    def get_hist_array(
        self, field_hist_template, 
        category=None, cuts=None, multi_proc=False):
        """
        """
        sel = self.cuts(category)
        if not cuts is None:
            sel &= cuts
        field_hists = {}
        self.branches = field_hist_template.keys()
        log.debug('Will access branches: {0}'.format(self.branches))
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

    def get_2d_map(
        self, var1, var2, prefix='off',
        category=None, cuts=None, dummy_range=False):
        """
        """
        sel = self.cuts(category)

        if cuts is not None:
            sel &= cuts

        nbins1, xmin1, xmax1 = var1['bins'], var1['range'][0], var1['range'][1]
        nbins2, xmin2, xmax2 = var2['bins'], var2['range'][0], var2['range'][1]
        
        if dummy_range:
            nbins1, xmin1, xmax1 = 40, -1e5, 1e5
            nbins2, xmin2, xmax2 = 40, -1e5, 1e5
            

        if 'prefix' in var1:
            if not isinstance(var1['prefix'], (list, tuple)):
                var1['prefix'] = [var1['prefix']]

            if prefix in var1['prefix']:
                var1_expr = prefix + '_' + var1['name']
            else:
                var1_expr = var1['name']
        else:
            var1_expr = var1['name']
            

        if 'scale' in var1:
            var1_expr = '{0}*{1}'.format(var1_expr, var1['scale'])


        if 'prefix' in var2:
            if not isinstance(var2['prefix'], (list, tuple)):
                var2['prefix'] = [var2['prefix']]

            if prefix in var2['prefix']:
                var2_expr = prefix + '_' + var2['name']
            else:
                var2_expr = var2['name']
        else:
            var2_expr = var2['name']
            

        if 'scale' in var2:
            var2_expr = '{0}*{1}'.format(var2_expr, var2['scale'])

        expr = '{0}:{1}'.format(var1_expr, var2_expr)
        log.debug(expr)
        hist= Hist2D(
            nbins1, xmin1, xmax1,
            nbins2, xmin2, xmax2, 
            type='D')
        hist = self.draw_helper(hist, expr, sel)
        return hist
