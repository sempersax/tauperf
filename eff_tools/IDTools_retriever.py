from eff_tools import IDTools_handler 
from eff_tools.TauIDTool      import TauIDTool


# --> Declaration of the different id tools
def get_IDTools(tauCell):
    ID_Tools = {}
    ID_Tools["bdt_presel_3var"] = TauIDTool(tauCell,IDTools_handler.inputs_lists["bdt_presel_3var"])
    ID_Tools["bdt_presel_5var"] = TauIDTool(tauCell,IDTools_handler.inputs_lists["bdt_presel_5var"])
    ID_Tools["bdt_presel_fullvarlist"] = TauIDTool(tauCell,IDTools_handler.inputs_lists["bdt_presel_fullvarlist"])
    ID_Tools["bdt_presel_fullvarlist_michel1"] = TauIDTool(tauCell,IDTools_handler.inputs_lists["bdt_presel_fullvarlist_michel1"])
    ID_Tools["bdt_presel_fullvarlist_michel2"] = TauIDTool(tauCell,IDTools_handler.inputs_lists["bdt_presel_fullvarlist_michel2"])
    ID_Tools["bdt_presel_fullvarlist_michel3"] = TauIDTool(tauCell,IDTools_handler.inputs_lists["bdt_presel_fullvarlist_michel3"])
    ID_Tools["bdt_full_quentin_new"]           = TauIDTool(tauCell,IDTools_handler.inputs_lists["bdt_full_quentin_new"])
    cutvals = IDTools_handler.cutvals
    return ID_Tools,cutvals
