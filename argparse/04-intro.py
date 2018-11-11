import argparse
# short versions of options:
parser = argparse.ArgumentParser()
parser.add_argument('-v',
                    '--verbose',
                    help='increase verbosity',
                    action='store_true')
args = parser.parse_args()
if args.verbose:
    print('verbosity turned on!')

'''run: python 04-intro.py -v '''