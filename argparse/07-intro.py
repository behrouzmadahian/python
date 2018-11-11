import argparse
'''
Sometimes a script may only parse a few of the command-line arguments, passing the remaining arguments on to another 
script or program. In these cases, the parse_known_args() method can be useful. It works much like parse_args() 
except that it does not produce an error when extra arguments are present. Instead, it returns a two item tuple 
containing the populated namespace and the list of remaining argument strings.
'''
parser = argparse.ArgumentParser()
parser.add_argument('--foo', action='store_true')
parser.add_argument('bar')
args = parser.parse_known_args(['--foo', '--badger', 'BAR', 'spam'])
print(args)