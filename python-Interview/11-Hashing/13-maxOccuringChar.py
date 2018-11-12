'''
Return maximum occurring character in an input string
'''
def maxChar(st):
    ht ={}
    for ch in st:
        if ch in ht:
            ht[ch] += 1
        else:
            ht[ch] = 1
    keys = list(ht.keys())
    print(keys)
    maxCnt, maxCh = ht[keys[0]],keys[0]
    for key in keys[1:]:
        if ht[key] > maxCnt:
            maxCnt, maxCh = ht[key], key
    return maxCnt, maxCh
st = 'hi this h'
print(maxChar(st))