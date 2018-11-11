import argparse
'''
The 02- example accepts arbitrary integer values for --verbosity, but for our simple program,
 only two values are actually useful, True or False. Let’s modify the code accordingly:
 Note that we now specify a new keyword, action, and give it the value "store_true".
  This means that, if the option is specified, assign the value True to args.verbose. Not specifying it implies False.
  right now verbose is a flag, if we add it to command line it means True!
'''
# optional arguments
parser = argparse.ArgumentParser()
parser.add_argument('--verbose',
                    help='increase output verbosity',
                    action='store_true')
parser.add_argument('--var',
                    help='a variable to square',
                    type=int)
args = parser.parse_args()
if args.verbose:
    print('Verbosity turned on!')
print('Square= ', args.var**2)
'''run: python 03-intro.py --verbose --var 3'''