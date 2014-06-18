# --> python imports
from array import array
# --> ROOT/rootpy imports
from ROOT import TMVA
from rootpy.extern import ordereddict
# local imports
from . import log; log = log[__name__]

class DecisionTool(object):
    """
    TODO: add description
    """
    def __init__(self,
                 name,
                 weight_file,
                 variables,
                 cutval,
                 training = 'training'):
        """ A class to handle the decision of the BDT"""
        TMVA.Tools.Instance()
        self._reader = TMVA.Reader()
        self._vars = variables
        self._vals = [array('f', [0.]) for i in range(0, len(variables))] 
        self._cutval = cutval
        self._score = -9999
        self._name = name
        self._training = training
        log.info('SetReader: {0}, {1}, {2}'.format(name, weight_file, variables))
        for var, val in zip(self._vars, self._vals):
            self._reader.AddVariable(var[self._training], val)
        self._reader.BookMVA(name, weight_file)


    @property
    def cutval(self):
        return self._cutval
    
    @cutval.setter
    def cutval(self, cutval):
        self._cutval = cutval

    @property
    def score(self):
        return self._score

    def Evaluate(self, tau):
        for var, val in zip(self._vars, self._vals):
            val[0] = getattr(tau, var['name'])
            log.info('{0}: {1}'.format(var['name'], val[0]))
        self.score = self._reader.EvaluateMVA(self._name)

    # --------------------------------------------
    def Decision(self, tau):
        self.Evaluate(tau)
        log.debug('BDT: {0} - {1} - {2}'.format(self.cutval, self._name, self.score))
        if self.score>=self.cutval:
            return True
        else:
            return False
    
