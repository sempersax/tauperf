# ROOT/rootpy imports
from rootpy.extern import argparse

#--> Receive and parse argument
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_list",
                        help="the list (txt file) of the input files")
    parser.add_argument("output_file",
                        help="the name of the output root file")
    parser.add_argument('--signal', action='store_true', default=False,
                        help="Sample type signal/bkg")
    parser.add_argument("-N","--Nentries", type=int, default=-1,
                        help="Specify the number of events use to run")

    return parser
