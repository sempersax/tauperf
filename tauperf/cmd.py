from argparse import ArgumentParser






def get_parser(action=None):

    parser = ArgumentParser()
    parser.add_argument(
        '--use-dy', default=False, 
        action='store_true', 
        help='Use DY simulation for signal')
    parser.add_argument(
        '--use-jz', default=False, 
        action='store_true', 
        help='Use JZ simulation for background')
    parser.add_argument('--categories', default='plotting')
    parser.add_argument('--var', default=None, nargs='*', help='Specify a particular variable')
    parser.add_argument('--cut', default=None, type=str, help='additional cut to apply')
    parser.add_argument('--level', default='off', type=str, choices=['off', 'hlt'], help='additional cut to apply')
    parser.add_argument('--trigger', default=False, action='store_true')
    parser.add_argument('--no-weight', default=False, action='store_true')
    parser.add_argument('--features', default='features_pileup_corrected', type=str)
    parser.add_argument('--verbose', default=False, action='store_true')
    if action == 'plot':
        parser.add_argument('--logy', default=False, action='store_true')

    return parser
