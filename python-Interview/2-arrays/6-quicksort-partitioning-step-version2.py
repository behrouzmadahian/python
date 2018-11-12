'''
keep 3 indicies: smaller, equal, larger.
and decide on where an element need to go in a single pass
index through equal and decide
'''
def dutch_flag_partition2(a, index):
    pivot = a[index]
    smaller, equal, larger = 0, 0, len(a) - 1

    while equal < larger:
        if a[equal] < pivot:
            a[smaller], a[equal] = a[equal], a[smaller]
            smaller += 1
            equal += 1

        # if equal to pivot dont' do anything. go to next element
        if a[equal] == pivot:
            equal += 1

        if a[equal] > pivot:
            a[larger], a[equal] = a[equal], a[larger]
            larger -= 1
    return a
print(dutch_flag_partition2([2,5, 3, 2, 8,0], 3))


