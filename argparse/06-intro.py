import argparse
# restricting the values a argument can take:
parser = argparse.ArgumentParser()
parser.add_argument('--verbose',
                    help='increase verbosity',
                    choices=[0, 1, 2],
                    type=int)
parser.add_argument('square',
                    help='argument to square',
                    type=int)
args = parser.parse_args()
ans = args.square**2
if args.verbose == 2:
    print('the square of {} is {}'.format(args.square, ans))
elif args.verbose == 1:
    print('{}^2 == {}'.format(args.square, ans))
else:
    print(ans)