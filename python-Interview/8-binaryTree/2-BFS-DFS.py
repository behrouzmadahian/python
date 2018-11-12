'''
A Tree is typically traversed in two ways:
    Breadth First Traversal (Or Level Order Traversal)
    Depth First Traversals
DFS:
    Inorder Traversal (Left-Root-Right)
    Preorder Traversal (Root-Left-Right)
    Postorder Traversal (Left-Right-Root)

All four traversals require O(n) time as they visit every node exactly once.
Is there any difference in terms of Extra Space?
There is difference in terms of extra space required.
1- Extra Space required for Level Order Traversal is O(w) where w is maximum width of Binary Tree.
2- Extra Space required for Depth First Traversals is O(h) where h is maximum height of Binary Tree.

It is evident from above points that extra space required for Level order traversal is likely to be more
when tree is more balanced and extra space for Depth First Traversal is likely to be more when tree is less balanced.

Depth First Traversals are typically recursive and recursive code requires function call overheads.
The most important points is, BFS starts visiting nodes from root while DFS starts visiting nodes from leaves.
So if our problem is to search something that is more likely to closer to root, we would prefer BFS.
And if the target node is close to a leaf, we would prefer DFS.
'''