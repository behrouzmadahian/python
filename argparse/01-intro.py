import argparse
'''
parser for command-line options, arguments and sub commands.
It makes it easy to write user-friendly command line interfaces.
The program defines what arguments it requires, and argparse will figure out how
to parse those out of sys.argv. 
The argparse module also automatically generates help and usage messages and
issues errors when users give the program invalid arguments.
argparse treats the argument we define in it as STRING! thus explicitly define the argument type!!

'''
parser = argparse.ArgumentParser()
# go to command line and write:python 01-intro.py --help
# As seen above, even though we didnt specify any help arguments in our
# script, its still giving us a nice help message.

# 1. Positional Arguments
# Whenever we want to specify which command-line options the program will accept,
# we use the "add_argument()" method.
parser.add_argument('echo',
                    help='echo the string you use here',
                    type=int)  # naming the argument echo
args = parser.parse_args()   # returns data from the options specified
print(args.echo**2)

