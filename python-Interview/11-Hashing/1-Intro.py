'''
With hashing we get O(1) search time / insert/ delete on average (under reasonable assumptions) and O(n) in worst case.
assume we want a structure that to search for employee information  using their phone number.
The idea is to use hash function that converts a given phone number or any other key to a smaller
number and uses the small number as index in a table called hash table.

Hash Function:
 A function that converts a given big phone number to a small practical integer value.
 The mapped integer value is used as an index in hash table.
 In simple terms, a hash function maps a big number or string to a small
 integer that can be used as index in hash table.
A good hash function should have following properties
1) Efficiently computable.
2) Should uniformly distribute the keys (Each table position equally likely for each key)
For example for phone numbers a bad hash function is to take first three digits.
A better function is consider last three digits.
Please note that this may not be the best hash function. There may be better ways.

Hash Table: An array that stores pointers to records corresponding to a given phone number.
An entry in hash table is None if no existing phone number has hash function value equal to the index for the entry.
'''