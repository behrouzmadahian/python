import argparse
# 2- Optional arguments!
parser = argparse.ArgumentParser()
parser.add_argument('--verbosity',
                    help='increase output verbosity')
args = parser.parse_args()
if args.verbosity:
    print('verbosity turned on!')

'''
from commandline:
python 02-intro2.py --verbosity 1

The above example accepts arbitrary integer values for --verbosity, but for our simple program, only two values
 are actually useful, True or False. Let’s modify the code accordingly:
'''
