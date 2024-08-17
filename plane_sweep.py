from utilites import *
import matplotlib.pyplot as plt
''' put this file and utilites.py in the same dir '' 

our_tree = AVLTree()
our_pqueue = MinHeap() # NEW
is_in_pqueue = AVLTree()

def plane_sweep(our_tree, our_pqueue): # after i pushed all points to pqueue
    is_in_pqueue = None # nullify it (the rest are already nullified
    is_in_pqueue = AVLTree() # nullify it (the rest are already nullified
    counter = 0
    while (len(our_pqueue.heap)!=0):  # NEW
        p = our_pqueue.extractMin()  # NEW
        x_cor = p.x
        if(p.type == START):
            seg = p.first_segment
            key = (seg.a(),seg.b()) #(m,b)
            our_tree.root = our_tree.insert_node(our_tree.root,key,seg,x_cor)
            successor, predecessor = our_tree.inorder_successor_predecessor(our_tree.root, key, x_cor)
            if (successor is not None):
                successor = successor.data
            if (predecessor is not None):
                predecessor = predecessor.data
            if (intersects(seg, predecessor) and predecessor is not None):
                intersection_p = intersection(predecessor,seg)
                our_pqueue.insertKey(intersection_p) #NEW
                is_in_pqueue.root = is_in_pqueue.insert_node_pq(is_in_pqueue.root,(seg.id,predecessor.id))
            if (intersects(seg, successor) and successor is not None):
                intersection_p = intersection(seg, successor)
                our_pqueue.insertKey(intersection_p) #NEW
                is_in_pqueue.root = is_in_pqueue.insert_node_pq(is_in_pqueue.root,(seg.id,successor.id))
        elif(p.type == END):
            seg = p.first_segment
            key = (seg.a(),seg.b()) #(m,b)
            successor, predecessor = our_tree.inorder_successor_predecessor(our_tree.root, key, x_cor)
            if (successor is not None):
                successor = successor.data
            if (predecessor is not None):
                predecessor = predecessor.data
            our_tree.root = our_tree.delete_node(our_tree.root,key,x_cor)
            if (successor is not None and predecessor is not None):
                if (intersects(successor, predecessor) and not is_in_pqueue.key_exists((successor.id,predecessor.id))):
                    intersection_p = intersection(predecessor,successor)
                    our_pqueue.insertKey(intersection_p)  # NEW
                    is_in_pqueue.root = is_in_pqueue.insert_node_pq(is_in_pqueue.root, (successor.id, predecessor.id))
        else: #p.type == INTERSECTION
            seg1,seg2 = None, None
            seg1 = p.first_segment
            seg2 = p.second_segment
            key1 = (seg1.a(),seg1.b()) #(m,b)
            key2 = (seg2.a(),seg2.b()) #(m,b)

            node1, node2 = None, None
            node1,node2 = our_tree.get_node(our_tree.root,key1,x_cor)

            if (abs(node1.key[0]-key1[0])>EPSILON): # certain structure of the tree
                successor1,successor2,predecessor1,predecessor2 = None, None,None,None
                successor1, predecessor1 = our_tree.inorder_successor_predecessor(our_tree.root, key1, x_cor)
                successor2, predecessor2 = our_tree.inorder_successor_predecessor(our_tree.root, key1, x_cor, True,False)
                if (successor1 is not None):
                    successor1 = successor1.data
                if (predecessor1 is not None):
                    predecessor1 = predecessor1.data
                if (successor2 is not None):
                    successor2 = successor2.data
                if (predecessor2 is not None):
                    predecessor2 = predecessor2.data

                ## swaps here keys and data
                node1.data = seg1
                node2.data = seg2
                node1.key = key1
                node2.key = key2


                #checking intersection with new succcer and predecessor
                if (successor1 is not None):
                    if (intersects(successor1, seg1) and not is_in_pqueue.key_exists((successor1.id, seg1.id))):
                        intersection_p = intersection(seg1,successor1)
                        our_pqueue.insertKey(intersection_p)
                        is_in_pqueue.root = is_in_pqueue.insert_node_pq(is_in_pqueue.root, (successor1.id, seg1.id))

                if (predecessor2 is not None):
                    if (intersects(predecessor2, seg2) and not is_in_pqueue.key_exists((predecessor2.id, seg2.id))):
                        intersection_p = intersection(predecessor2, seg2)
                        our_pqueue.insertKey(intersection_p)
                        is_in_pqueue.root = is_in_pqueue.insert_node_pq(is_in_pqueue.root, (predecessor2.id, seg2.id))
            else: # other structure of the tree

                #reverse down here
                successor1,successor2,predecessor1,predecessor2 = None, None,None,None
                successor1, predecessor1 = our_tree.inorder_successor_predecessor(our_tree.root, key1, x_cor)
                successor2, predecessor2 = our_tree.inorder_successor_predecessor(our_tree.root, key1, x_cor, False,True)
                if (successor1 is not None):
                    successor1 = successor1.data
                if (predecessor1 is not None):
                    predecessor1 = predecessor1.data
                if (successor2 is not None):
                    successor2 = successor2.data
                if (predecessor2 is not None):
                    predecessor2 = predecessor2.data


                node1.data = seg2
                node2.data = seg1
                node1.key = key2
                node2.key = key1


                if (successor2 is not None):
                    if (intersects(successor2, seg1) and not is_in_pqueue.key_exists((successor2.id,seg1.id))):
                        intersection_p = intersection(seg1, successor2)
                        our_pqueue.insertKey(intersection_p)
                        is_in_pqueue.root = is_in_pqueue.insert_node_pq(is_in_pqueue.root, (successor2.id, seg1.id))

                if (predecessor1 is not None):
                    if (intersects(predecessor1, seg2) and not is_in_pqueue.key_exists((predecessor1.id,seg2.id))):
                        intersection_p = intersection(predecessor1, seg2)
                        our_pqueue.insertKey(intersection_p)
                        is_in_pqueue.root = is_in_pqueue.insert_node_pq(is_in_pqueue.root, (predecessor1.id, seg2.id))

            counter +=1
    print(counter)

def get_file_and_sweep(filepath):

    # Open the ASCII file for reading
    with open(filepath, 'r') as file:
        # Flag that tracks if the first line has been processed
        first_line_processed = False
        num_of_segments_processed = False
        id = 0
        for line in file:
            if not first_line_processed: # beginning - number of sets
                values = line.split()
                nums = [float(value) for value in values if value.strip()]
                for num in nums:
                    num_of_test_cases = num
                # Setting the flag to True to indicate that the first line has been processed
                first_line_processed = True
                continue

            if not num_of_segments_processed: # begining of a set - number of segments
                values = line.split()
                nums = [float(value) for value in values if value.strip()]
                for num in nums:
                    num_of_segments = num
                if(num_of_segments ==-1): #EOF
                    break
                # Setting the flag to True to indicate that the first line has been processed
                num_of_segments_processed = True
                our_tree = AVLTree()
                our_pqueue = MinHeap()

                segments = []  # debug

                continue

            p , q, seg= None,None, None
            if (num_of_segments>0): # process every segment
                # Processing the rest of the lines
                values = line.split()
                nums = [float(value) for value in values if value.strip()]
                i = 0
                if(nums[i]<nums[i+2]):
                    p= Point(nums[i],nums[i+1],START)
                    q = Point(nums[i+2],nums[i+3],END)
                else:
                    p = Point(nums[i],nums[i+1],END)
                    q = Point(nums[i+2],nums[i+3],START)
                seg = Segment(id,p,q)
                p.first_segment = seg
                q.first_segment = seg
                segments.append(seg)  # debug
                our_pqueue.insertKey(p)
                our_pqueue.insertKey(q)
                id+=1
                num_of_segments-=1

            ## ! entering here to pqueue

            if(num_of_segments==0) : # processed all segments we reached num_of_segments_processed
                num_of_segments_processed=False
                plane_sweep(our_tree, our_pqueue)

get_file_and_sweep('your_file.txt')
#  UP HERE PUT THE NAME OF THE ASCII FILE YOU WANT TO RUN THAT IN THE SAME DIR HAS THE TWO FILES IN THIS ZIP FILE.
