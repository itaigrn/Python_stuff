START = 0
END = 1
INTERSECTION = 2
EPSILON=0.00000000001

import matplotlib.pyplot as plt
import networkx as nx


class Point:
    x: float  # float
    y: float  # float
    type: int # one from the above MACROS
    first_segment_id: int
    second_segment_id: int

    def __init__(self, x=0, y=0,type=3, first_segment=None, second_segment=None):
        self.x = x
        self.y = y
        self.type = type
        self.first_segment = first_segment
        self.second_segment = second_segment

    def __str__(self):
        return f"({self.x}, {self.y})"

class Segment:
    id: int  # to overcome numerical error when we find a point on an
    #    # already-known segment we identify segments with unique ID.
    #    # binary search with numerical errors is guaranteed to find an
    #    # index whose distance from the correct one is O(1) (here it is 2).
    p: Point  # Point, after input we compare and swap to guarantee that p.x <= q.x
    q: Point  # Point

    def __init__(self, id, p, q):  # ITAI: I added the id argument
        if p.x > q.x:
            p, q = q, p
        self.p = p
        self.q = q
        self.id = id # ITAI ADDED

    def a(self):  # () -> double
        return ((self.p.y - self.q.y) / (self.p.x - self.q.x))

    def b(self):  # () -> double
        return (self.p.y - (self.a() * self.p.x))

    # the y-coordinate of the point on the segment whose x-coordinate
    #   is given. Segment boundaries are NOT enforced here.
    def calc(x):
        return (self.a() * (x + self.b()))



# class

def is_left_turn(a, b, c):  # (Point,Point,Point) -> bool
    x1 = a.x
    x2 = b.x
    x3 = c.x
    y1 = a.y
    y2 = b.y
    y3 = c.y
    return ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))) > 0


def intersection(s1, s2):  # (segment,segment) -> Point | None
    if (s1 is None or s2 is None): return None

    if ((is_left_turn(s1.p, s1.q, s2.p) != is_left_turn(s1.p, s1.q, s2.q)) and
            (is_left_turn(s2.p, s2.q, s1.p) != is_left_turn(s2.p, s2.q, s1.q))):

        a1 = s1.a()
        a2 = s2.a()

        b1 = s1.b()
        b2 = s2.b()

        # commutation consistency: sort by a (then by b) # ITAI: don't understand why it is necessary
        if a1 > a2 or (a1 == a2 and b1 > b2):
            a1, a2 = a2, a1
            b1, b2 = b2, b1

        x = (b2 - b1) / (a1 - a2)
        y = a1 * x + b1

        return Point(x, y,INTERSECTION,s1,s2)
    else:
        return None;


def intersects(s1, s2):  # (Segment,Segment) -> bool
    return not (intersection(s1, s2) is None)



class CG24PriorityQueue:
    max1: bool  # bool
    max2: bool  # bool
    max3: bool  # bool
    t: int  # int
    arr: any  # any[]

    class cEntry:
        def __init__(self):
            pass

    def __init__(self, priorityMax=True, tiebreakerMax=True, tiebreaker2Max=True):
        self.max1 = priorityMax
        self.max2 = tiebreakerMax
        self.max3 = tiebreaker2Max
        self.t = int(0)
        self.arr = list()


    def compare(self, l, r):  # (p1,p2) -> bool
        if l.p != r.p:
            return (l.p > r.p) if self.max1 else (l.p < r.p)
        if l.p2 != r.p2:
            return (l.p2 > r.p2) if self.max2 else (l.p2 < r.p2)
        if l.p != r.p:
            return (l.p3 > r.p3) if self.max3 else (l.p3 < r.p3)
        return l.pzm < r.pzm;


    def insert(self, data, p, tiebreaker=0, tiebreaker2=0):  # (any, double[, double[, double]]) -> void
        entry = CG24PriorityQueue.cEntry()
        entry.p = float(p)
        entry.p2 = float(tiebreaker)
        entry.p3 = float(tiebreaker2)
        entry.pzm = self.t
        entry.data = data

        self.t = self.t + int(1)
        self.arr.append(entry)
        # heapify up
        i = int(len(self.arr)) - int(1)
        parent = int(i / 2)
        while i != parent and self.compare(self.arr[i], self.arr[parent]):
            self.arr[i], self.arr[parent] = self.arr[parent], self.arr[i]
            i = parent
            parent = int(i / 2)

    def empty(self):  # () -> bool
        return 0 == len(self.arr)

    def pop(self):  # () -> any
        if 0 == len(self.arr):
            return self.arr[0]  # raise exception

        res = self.arr[0].data

        if len(self.arr) > 1:
            n = len(self.arr)
            self.arr[0], self.arr[n - 1] = self.arr[n - 1], self.arr[0]
            n = n - 1
            i = 0
            while i < n:
                best = i
                j1 = int(2 * i + 1)
                j2 = int(2 * i + 2)
                if j1 < n and self.compare(self.arr[j1], self.arr[best]):
                    best = j1
                if j2 < n and self.compare(self.arr[j2], self.arr[best]):
                    best = j2
                if best == i:
                    break
                self.arr[i], self.arr[best] = self.arr[best], self.arr[i]
                i = best

        self.arr.pop()
        return res

import sys

# Create a tree node
class TreeNode(object):
    def __init__(self, key,data=None): # seg=None from when I use is_in_pqueue tree
        self.key = key # $$ needs to be of a form (m,b)
        self.left = None
        self.right = None
        self.height = 1
        self.data = data # can be either seg, or point

class AVLTree(object):
    def __init__(self):
        self.root = None
    def calc_key_y(self,key,x_cor):
        m = key[0]
        b = key[1]
        return m*x_cor+b

    # Function to insert a node
    def insert_node(self, root, key,seg, x_cor): # ITAI: added x_cor
        # Find the correct location and insert the node
        if not root:
            return TreeNode(key,seg)
        #elif key < root.key:
        elif self.calc_key_y(key, x_cor)- self.calc_key_y(root.key, x_cor)< -EPSILON:
            root.left = self.insert_node(root.left, key,seg,x_cor)
        else: # whether it is greater or equal!!
            root.right = self.insert_node(root.right, key,seg,x_cor)

        root.height = 1 + max(self.getHeight(root.left),self.getHeight(root.right))

        # Update the balance factor and balance the tree
        balanceFactor = self.getBalance(root)
        if balanceFactor > 1:
            #if key < root.left.key:
            if self.calc_key_y(key, x_cor) - self.calc_key_y(root.left.key, x_cor) < -EPSILON:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)

        if balanceFactor < -1:
            #if key > root.right.key:
            if self.calc_key_y(key, x_cor)-self.calc_key_y(root.right.key, x_cor) > EPSILON:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)

        return root

    # Function to delete a node
    def delete_node(self, root, key, x_cor): # ITAI: added x_cor

        if not root:
            return root
        #elif key < root.key:
        elif self.calc_key_y(key, x_cor) - self.calc_key_y(root.key, x_cor) < -EPSILON:
            root.left = self.delete_node(root.left, key,x_cor)
        #elif key > root.key:
        elif self.calc_key_y(key, x_cor) - self.calc_key_y(root.key, x_cor) > EPSILON:
            root.right = self.delete_node(root.right, key,x_cor)
        else:
            if root.left is None:
                temp = root.right
                root = None
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                return temp
            temp = self.getMinValueNode(root.right)
            root.key = temp.key
            root.data = temp.data ## NEW - ITAIs
            root.right = self.delete_node(root.right,temp.key,x_cor)
        if root is None:
            return root

        # Update the balance factor of nodes
        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        balanceFactor = self.getBalance(root)

        # Balance the tree
        if balanceFactor > 1:
            if self.getBalance(root.left) >= 0:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)
        if balanceFactor < -1:
            if self.getBalance(root.right) <= 0:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)
        return root

    # Function to perform left rotation
    def leftRotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.getHeight(z.left),
                           self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                           self.getHeight(y.right))
        return y

    # Function to perform right rotation
    def rightRotate(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.getHeight(z.left),
                           self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                           self.getHeight(y.right))
        return y

    # Get the height of the node
    def getHeight(self, root):
        if not root:
            return 0
        return root.height

    # Get balance factore of the node
    def getBalance(self, root):
        if not root:
            return 0
        return self.getHeight(root.left) - self.getHeight(root.right)

    def getMinValueNode(self, root):
        if root is None or root.left is None:
            return root
        return self.getMinValueNode(root.left)

    def preOrder(self, root):
        if not root:
            return
        print("{0} ".format(root.key), end="")
        self.preOrder(root.left)
        self.preOrder(root.right)

    # Print the tree
    def printHelper(self, currPtr, indent, last):
        if currPtr != None:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "
            print(currPtr.key)
            self.printHelper(currPtr.left, indent, False)
            self.printHelper(currPtr.right, indent, True)

    def find_max(self, node): # need to check this function
        if node is None:
            return None

        while node.right is not None:
            node = node.right

        return node

    def find_min(self, node): # need to check this function
        if node is None:
            return None

        while node.left is not None:
            node = node.left

        return node

    def inorder_successor_predecessor(self, root, key, x_cor,found_first_left=False,found_first_right=False): # need to check this function
        if root is None:
            return None, None

        successor = None
        predecessor = None

        while root is not None:
            if self.calc_key_y(key, x_cor) - self.calc_key_y(root.key, x_cor) < -EPSILON:
                successor = root
                root = root.left
            elif self.calc_key_y(key, x_cor) - self.calc_key_y(root.key, x_cor) > EPSILON:
                predecessor = root
                root = root.right
            elif found_first_left: #eaual but stil I go left!! (and only left)
                found_first_left = False
                successor = root
                root = root.left
            elif found_first_right:  # eaual but stil I go right!! (and only right)
                found_first_right = False
                predecessor = root
                root = root.right

            else:
                if root.left is not None:
                    predecessor = self.find_max(root.left)
                if root.right is not None:
                    successor = self.find_min(root.right)
                break

        return successor, predecessor

    def get_node(self, root, key,x_cor): # always for to
        nodes =[]
        self.get_node_aux(root, key, x_cor,nodes)
        assert(len(nodes)==2)
        return nodes
    def get_node_aux(self, root, key,x_cor,nodes):
        # Base case: If the root is None, the key is not found
        if root is None:
            return #None

        if (abs(self.calc_key_y(key, x_cor) - self.calc_key_y(root.key, x_cor)) < EPSILON):
            nodes.append(root)

        if (self.calc_key_y(key, x_cor) - self.calc_key_y(root.key, x_cor) > EPSILON) :
            self.get_node_aux(root.right, key, x_cor,nodes)

        elif (self.calc_key_y(key, x_cor) - self.calc_key_y(root.key, x_cor) < -EPSILON):
            #return root
            self.get_node_aux(root.left, key, x_cor,nodes)

        elif(len(nodes) < 2):
            # key is equal and I check both subtrees
            self.get_node_aux(root.left, key, x_cor, nodes)
            self.get_node_aux(root.right, key, x_cor, nodes)

    def insert_node_pq(self, root, key):
        if key[1] < key[0]:
            key = (key[1], key[0])
        if not root:
            return TreeNode(key)
        elif key < root.key:
            root.left = self.insert_node_pq(root.left, key)
        else:
            root.right = self.insert_node_pq(root.right, key)

        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        balanceFactor = self.getBalance(root)
        if balanceFactor > 1:
            if key< root.left.key:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)

        if balanceFactor < -1:
            if key> root.right.key:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)

        return root

    def delete_node_pq(self, root,key):
        if key[1] < key[0]:
            key = (key[1], key[0])

        if not root:
            return root
        elif key < root.key:
            root.left = self.delete_node_pq(root.left, key)
        elif key > root.key:
            root.right = self.delete_node_pq(root.right, key)
        else:
            if root.left is None:
                temp = root.right
                root = None
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                return temp
            temp = self.getMinValueNode(root.right)
            root.key = temp.key
            root.right = self.delete_node_pq(root.right,
                                          temp.key)
        if root is None:
            return root

        # Update the balance factor of nodes
        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        balanceFactor = self.getBalance(root)

        # Balance the tree
        if balanceFactor > 1:
            if self.getBalance(root.left) >= 0:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)
        if balanceFactor < -1:
            if self.getBalance(root.right) <= 0:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)
        return root

    def search_key(self, root, key, x_cor):
        if root is None:
            return False


        if (self.calc_key_y(key,x_cor)-self.calc_key_y(key,x_cor) < EPSILON):
            return True

        elif (self.calc_key_y(key,x_cor)-self.calc_key_y(root.key)<-EPSILON):
            return self.search_key(root.left, key)
        # If the key is greater than the current key
        else:
            return self.search_key(root.right, key)

    def search_key_pq(self, root, key):
        if root is None:
            return False

        if key == root.key:
            return True
        elif key < root.key:
            return self.search_key_pq(root.left, key)
        # If the key is greater than the current  key
        else:
            return self.search_key_pq(root.right, key)

    def key_exists(self, key):
        if key[0]<key[1]:
            return self.search_key_pq(self.root, key)  # Assuming root is a member variable
        else:
            ordered_key = (key[1],key[0])
            return self.search_key_pq(self.root, ordered_key)



# 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
# is the index of a leaf with a possibly out-of-order value.  Restore the
# heap invariant.
def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem.x - parent.x < -EPSILON:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos].x - heap[rightpos].x < -EPSILON:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)
def heappush(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)

def heappop(heap):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt

def heapify(x):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    n = len(x)
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in reversed(range(n//2)):
        _siftup(x, i)



# heappop - pop and return the smallest element from heap
# heappush - push the value item onto the heap, maintaining
#             heap invarient
# heapify - transform list into heap, in place, in linear time

# A class for Min Heap
class MinHeap:
    # Constructor to initialize a heap
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2 # told to change to integer division

    # Inserts a new key 'k'
    def insertKey(self, k):
        heappush(self.heap, k)

        # Decrease value of key at index 'i' to new_val

    # It is assumed that new_val is smaller than heap[i]
    def decreaseKey(self, i, new_val):
        self.heap[i] = new_val
        while (i != 0 and self.heap[self.parent(i)] > self.heap[i]):
            self.heap[i], self.heap[self.parent(i)] = (
                self.heap[self.parent(i)], self.heap[i])

            # Method to remove minimum element from min heap

    def extractMin(self): # POP
        return heappop(self.heap)

        # This function deletes key at index i. It first reduces

    # value to minus infinite and then calls extractMin()
    def deleteKey(self, i):
        self.decreaseKey(i, float("-inf"))
        self.extractMin()

        # Get the minimum element from the heap

    def getMin(self): # TOP
        return self.heap[0]

