import numpy as np
import pandas as pd
'''
Our goal will be to parse the recipe data into ingredient lists, 
so we can quickly find a recipe based on some ingredients we have on hand.
'''
try:
    recipes = pd.read_json('recipeitems-latest.json')
except ValueError as e:
    print('Value Error: ', e)

'''
We get a ValueError mentioning that there is "trailing data." Searching
for the text of this error on the Internet,  it seems that it's due to using a 
file in which each line is itself a  valid JSON, but the full file is not.
Let's check if this interpretation is true:
'''
with open('recipeitems-latest.json') as f:
    line = f.readline()
    print(pd.read_json(line).shape)
# ITS TRUE!
'''
Yes, apparently each line is a valid JSON, so we'll need to string them together.
One way we can do this is to actually construct a string representation 
containing all these JSON entries, and then load the whole thing with pd.read_json
'''
with open('recipeitems-latest.json', 'r', encoding='UTF8') as f:
    # extract each line
    data = (line.strip() for line in f)
    # reformat to a string that is [string of each json comma separated]
    data_json = "[{0}]".format(','.join(data))
    print(data_json[-100:])

recipes = pd.read_json(data_json)
print(recipes.shape)
print(recipes.iloc[0])
'''
There is a lot of information there, but much of it is in a very messy form,
as is typical of data scraped from the Web. In particular, the ingredient 
list is in string format; we're going to have to carefully extract the 
information we're interested in. Let's start by taking a closer look at the ingredients:
'''
print('Length of each Receipe:\n')
print(recipes['ingredients'].str.len().head(), '\n')
print(recipes['ingredients'].str.len().describe(), '\n')
''''
The ingredient lists average ~250 characters long, with a minimum of 0 and a maximum of nearly 10,000 characters!
'''
print('recipe with longest ingredient list:')
print(recipes['name'][np.argmax(recipes['ingredients'].str.len())])
print('let"s see how many of the recipes are for breakfast food:')
print(recipes.description.str.contains('[Bb]reakfast')[:10])
print(recipes.description.str.contains('[Bb]reakfast').sum(), '\n')
print('How many of the recipes list cinnamon as an ingredient:')
print(recipes.ingredients.str.contains('[Cc]innamon').sum(), '\n')
print('We could even look to see whether any recipes misspell the ingredient as "cinamon":')
print(recipes.ingredients.str.contains('[Cc]inamon').sum())
print('-'*100)
print('A SIMPLE Recipe Recommender:')
'''
Let's go a bit further, and start working on a simple recipe recommendation system: 
given a list of ingredients, find a recipe that uses all those ingredients. 
'''
spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
              'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']
# We can then build a Boolean DataFrame consisting of True and False values,
#  indicating whether this ingredient appears in the list:
import re
spice_df = pd.DataFrame(dict((spice, recipes.ingredients.str.contains(spice, re.IGNORECASE))
                             for spice in spice_list))
print(spice_df.shape)
print(' find a recipe that uses parsley, paprika, and tarragon:')
selection = spice_df.query('parsley & paprika & tarragon')
print(len(selection))
print('====')
print(recipes.name[selection.index])
