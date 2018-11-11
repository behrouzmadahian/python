import argparse

# combining positional and optional arguyments:
parser =argparse.ArgumentParser()
parser.add_argument('square',
                    type=int,
                    help='display a square of a given number')
parser.add_argument('--verbose',
                    help='increase verbosity',
                    action='store_true')
args = parser.parse_args()
ans = args.square**2
if args.verbose:
    print('The square of {} equals {}'.format(args.square, ans))
