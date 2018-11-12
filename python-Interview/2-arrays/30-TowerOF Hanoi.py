'''
Tower of Hanoi is a mathematical puzzle where we have three rods and n disks. The objective of the puzzle
 is to move the entire stack to another rod, obeying the following simple rules:
1) Only one disk can be moved at a time.
2) Each move consists of taking the upper disk from one of the stacks and placing it on top of another stack
 i.e. a disk can only be moved if it is the uppermost disk on a stack.
3) No disk may be placed on top of a smaller disk.
'''


def TowerOfHanoi(n, from_rod, to_rod, aux_rod):
    if n == 1:
        print("Move disk 1 from rod", from_rod, "to rod", to_rod)
        return
    TowerOfHanoi(n - 1, from_rod, aux_rod, to_rod)
    print("Move disk", n, "from rod", from_rod, "to rod", to_rod)
    TowerOfHanoi(n - 1, aux_rod, to_rod, from_rod)


# Driver code
n = 2
TowerOfHanoi(n, 'A', 'C', 'B')
# A, C, B are the name of rods

def towerOFH(n, fromRod, toRod, auxRod):
    if n==1:
        print('move disk 1 from %s to %s'%(fromRod, toRod))
        return
    towerOFH(n-1, fromRod, auxRod, toRod)
    print('move disk %d from %s to %s' % (n,fromRod, toRod))
    towerOFH(n-1, auxRod, toRod, fromRod )
print('---')
towerOFH(n, 'A', 'C', 'B')
def towerH2(n, fromRod, toRod, auxRod):
    if n==1:
        print('move disk 1 from %s to %s' % (fromRod, toRod))
        return
    towerH2(n-1, fromRod, auxRod, toRod)
    print('Move disk %d from %s to %s'%(n, fromRod, toRod))
    towerH2(n-1, auxRod, toRod, fromRod)

