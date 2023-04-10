from __future__ import annotations
from typing import Iterable
from myFunctions import nearMatching, execute_this
from statistics import mean
from copy import deepcopy
import random
import math


# For Heaps
MAXHEAP, MINHEAP = True, False
swaps = 0


class Heap:
    __slots__ = ["keys", "k", "heapType"]

    def __init__(self, *values, heapType=None, kValue=2):
        self.heapType, self.k, self.keys = heapType, kValue, list()
        self.addElements(values)

    def __str__(self):
        return str(self.keys)

    def addElements(self, *values):
        for value in Heap._breaker(values):
            self.keys.append(value)
        self.buildHeap()

    @staticmethod
    def _breaker(toBreak):
        for value in toBreak:
            if isinstance(value, int) or isinstance(value, float):
                yield value
            elif isinstance(value, set) or isinstance(value, list) or isinstance(value, tuple):
                yield from Heap._breaker(value)
            else:
                raise TypeError(
                    f"{value} of type {type(value).__name__} is not supported.")

    def kids(self, index):
        for n in range(1, self.k+1):
            yield (self.k*index+n)

    def Heapify(self, index):
        if self.heapType is MAXHEAP:
            largest = index
            for kid in self.kids(index):
                if kid >= len(self.keys):
                    break
                if self.keys[kid] > self.keys[largest]:
                    largest = kid
            if largest != index:
                self.keys[largest], self.keys[index] = self.keys[index], self.keys[largest]
                self.Heapify(largest)
        elif self.heapType is MINHEAP:
            smallest = index
            for kid in self.kids(index):
                if kid >= len(self.keys):
                    break
                if self.keys[kid] < self.keys[smallest]:
                    smallest = kid
            if smallest != index:
                self.keys[smallest], self.keys[index] = self.keys[index], self.keys[smallest]
                global swaps
                swaps += 1
                self.Heapify(smallest)

    def buildHeap(self):
        if self.heapType is not None:
            for index in range(len(self.keys)//self.k, -1, -1):
                self.Heapify(index)

    def extract(self):
        if len(self.keys) == 0:
            return None
        temp = self.keys[0]
        del self.keys[0]
        self.Heapify(0)
        return (temp)

    def changeHeapType(self):
        if self.heapType is not None:
            self.heapType = not self.heapType
        else:
            raise Exception(
                "Can't change Heap when no heap type is assingned. Try using setHeapType function instead")

    def setHeapType(self, HeapType):
        if isinstance(HeapType, bool):
            self.heapType = HeapType
        else:
            raise TypeError(
                f"Entered type {type(HeapType)} is not supported. try writing maxHeap or minHeap")

    def sort(self):
        sortedkeys = list()
        if self.heapType is MINHEAP:
            while (temp := self.extract()) is not None:
                sortedkeys.append(temp)
        elif self.heapType is MAXHEAP:
            while (temp := self.extract()) is not None:
                sortedkeys.insert(0, temp)
        else:
            raise TypeError(
                "can't sort a heap when its type is not specified, try using list sort method instead")
        self.keys = sortedkeys
        del sortedkeys


class PriorityQueue:

    class item:
        __slots__ = ["priority", "task", 'id']

        def __init__(self, priority: int, task, idnum=None):
            self.priority, self.task, self.id = priority, task, idnum

        def __str__(self):
            return str({"Task": self.task, "Priority": self.priority, "ID": self.id})

        def __call__(self):
            return self

        def copy(self):
            return deepcopy(self)

    __slots__ = ['items', 'currTask', 'totalItems']

    def __init__(self, *args, priorityList=None, taskList=None):
        self.items, self.currTask, self.totalItems = list(), None, int()
        if priorityList is not None and taskList is not None:
            data = zip(priorityList, taskList)
            self.addItems(data)
        if any(args):
            self.addItems(args)

    def __str__(self):
        copyInstance, final = self.copy(), str()
        while any(copyInstance.items):
            final += str(copyInstance.getMostImportanTask().task) + '\n'
            copyInstance.markDone()
            # for x in copyInstance.items:
            # print(x)
        return final

    def addItems(self, Items):
        if isinstance(Items, dict):
            for x in Items.keys():
                self.items.append(
                    self.item(priority=x, task=Items[x], idnum=self.totalItems))
                self.totalItems += 1
        elif isinstance(Items, self.item):
            self.items.append(Items)
            self.totalItems += 1
        elif (isinstance(Items, tuple) or isinstance(Items, list)) and len(Items) == 2:
            self.items.append(
                self.item(priority=Items[0], task=Items[1], idnum=self.totalItems))
            self.totalItems += 1
        elif isinstance(Items, list) or isinstance(Items, tuple) or isinstance(Items, set):
            for item in Items:
                self.addItems(item)
        else:
            raise TypeError(
                f"item {Items} of type {type(Items).__name__} is not supported")
        self.buildQueue()

    def createItem(self, priority: int, task):
        self.addItems(self.item(priority, task))

    def kids(self, index):
        for n in range(1, 2+1):
            yield(2*index+n)

    def minHeapify(self, index):
        smallest = index
        for kid in self.kids(index):
            if kid >= self.totalItems:
                break
            if self.items[kid].priority < self.items[smallest].priority:
                smallest = kid
        if smallest != index:
            self.items[smallest].id, self.items[index].id = self.items[index].id, self.items[smallest].id
            self.items[smallest], self.items[index] = self.items[index], self.items[smallest]
            self.minHeapify(smallest)

    def buildQueue(self):
        for index in range(self.totalItems//2, -1, -1):
            self.minHeapify(index)

    def getTaskbyPriority(self, priority: int):
        for x in self.items:
            if x.priority == priority:
                self.currTask = x
                return x()

    def getTaskbyId(self, Idnum: int):
        self.currTask = self.items[Idnum]
        return self.items[Idnum]()

    def getMostImportanTask(self):
        self.currTask = self.items[0]
        return self.items[0]()

    def markDone(self, toMarkDone=None):
        if toMarkDone is not None and toMarkDone != self.currTask.id and isinstance(toMarkDone, int) and toMarkDone <= self.totalItems:
            for x in range(toMarkDone+1, self.totalItems):
                self.items[x].id -= 1
            del self.items[toMarkDone]
        elif isinstance(toMarkDone, self.item):
            for x in range(toMarkDone.id+1, self.totalItems):
                self.items[x].id -= 1
            del self.items[toMarkDone.id]
        elif toMarkDone is None:
            if self.currTask is None:
                self.currTask = self.getMostImportanTask()
            for x in range(self.currTask.id+1, self.totalItems):
                self.items[x].id -= 1
            del self.items[self.currTask.id]
        else:
            raise TypeError(
                f"passed argument {toMarkDone} of type {type(toMarkDone).__name__} is not supported")
        self.totalItems -= 1
        self.buildQueue()

    def copy(self):
        return deepcopy(self)


# For RedBlack Trees
RED, BLACK = False, True


class Node:
    __slots__ = ['parent', 'right', 'left', 'value', 'repititions']

    def __init__(self, parent=None, rightChild=None, leftChild=None, value=int(), repititions=int()):
        self.parent, self.right, self.left, self.value, self.repititions = parent, rightChild, leftChild, value, repititions

    def __str__(self):
        return str({'value': self.value, 'parent': self.parent.value if self.parent is not None else None, 'right': self.right.value if self.right is not None else None, 'left': self.left.value if self.left is not None else None, 'repititions': self.repititions})

    def __repr__(self):
        return NotImplemented

    def __call__(self):
        return (self)

    def __eq__(self, other):
        if self is not None and other is not None:
            if self.value == other.value:
                if self.right == other.right:
                    if self.left == other.left:
                        if self.repititions == other.repititions:
                            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        return deepcopy(self)

    @staticmethod
    def is_sentinel(node):
        return node is None or node == eval(str(type(node)).split('.')[-2]).sentinel


class BinarySearchTree:

    class leaf(Node):
        def __init__(self, parent=None, rightChild=None, leftChild=None, value=int(), repititions=int()):
            super().__init__(parent, rightChild, leftChild, value, repititions)

        def __str__(self):
            return str({'value': self.value, 'parent': self.parent.value if self.parent is not None else None, 'right': self.right.value if self.right is not None else None, 'left': self.left.value if self.left is not None else None, 'repititions': self.repititions})

    sentinel = leaf()

    def __init__(self, *args):
        elements = BinarySearchTree._breaker(args)
        index = nearMatching(elements, mean(elements))
        self.root = self.leaf(
            value=elements[index], parent=BinarySearchTree.sentinel)
        del elements[index]
        random.shuffle(elements)
        self.insert(elements)

    def __str__(self):
        returnString = str()
        indentCon = math.ceil(BinarySearchTree.height(self.root) / 1.5)

        def print_tree(node: BinarySearchTree.leaf, indent: int) -> None:
            if node is not None and not Node.is_sentinel(node):
                nonlocal returnString, indentCon
                print_tree(node.right, indent + indentCon)
                returnString += ' ' * indent + f"({node.value})\n"
                print_tree(node.left, indent + indentCon)
        print_tree(self.root, 0)
        return returnString

    def __repr__(self):
        return NotImplemented

    @staticmethod
    def _breaker(toBreak):
        final = list()
        for x in toBreak:
            if isinstance(x, int) or isinstance(x, float):
                final.append(x)
            elif isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set):
                final.extend(BinarySearchTree._breaker(x))
            else:
                raise TypeError(f"{x} of type {type(x)} isn't supported yet")
        return final

    def insert(self, *toAdd):
        newElements = eval(type(self).__name__)._breaker(toAdd)
        for num in newElements:
            iterator = self.root
            while True:
                if num == iterator.value:
                    iterator.repititions += 1
                    break
                elif num > iterator.value:
                    if not self.leaf.is_sentinel(iterator.right):
                        iterator = iterator.right
                    else:
                        iterator.right = self.leaf(parent=iterator, value=num)
                        break
                else:
                    if not self.leaf.is_sentinel(iterator.left):
                        iterator = iterator.left
                    else:
                        iterator.left = self.leaf(parent=iterator, value=num)
                        break

    def find(self, value, returnNode=False):
        iterator = self.root
        while not Node.is_sentinel(iterator):
            if value == iterator.value:
                if returnNode:
                    return iterator()
                return True
            if value > iterator.value:
                iterator = iterator.right
            else:
                iterator = iterator.left
        return False

    def iterGenerator(self, node):
        if node is None:
            return self.root
        if isinstance(node, self.leaf):
            return node
        elif isinstance(node, int):
            if (temp := self.find(node, returnNode=True)) is not False:
                return temp
            else:
                raise ValueError(f"value {node} doesn't exist in Tree")
        else:
            raise TypeError(
                f"object of type {type(node)} doesn't exist in tree")

    def max(self, tree=None, returnNode=False):
        iterator = self.iterGenerator(tree)
        while True:
            if Node.is_sentinel(iterator.right):
                if returnNode:
                    return iterator()
                return iterator.value
            iterator = iterator.right

    def min(self, tree=None, returnNode=False):
        iterator = self.iterGenerator(tree)
        while True:
            if Node.is_sentinel(iterator.left):
                if returnNode:
                    return iterator()
                return iterator.value
            iterator = iterator.left

    def next(self, value, returnNode=False):
        iterator = self.iterGenerator(value)
        if iterator.value == self.max():
            return None
        if Node.is_sentinel(iterator.right):
            return self.min(tree=iterator.right, returnNode=returnNode)
        while True:
            if iterator.parent.left is iterator:
                if returnNode:
                    return iterator.parent
                return iterator.parent.value
            iterator = iterator.parent

    def prev(self, value, returnNode=False):
        iterator = self.iterGenerator(value)
        if iterator.value == self.min():
            return None
        if Node.is_sentinel(iterator.left):
            return self.max(tree=iterator.left, returnNode=returnNode)
        while True:
            if iterator.parent.right is iterator:
                if returnNode:
                    return iterator.parent
                return iterator.parent.value
            iterator = iterator.parent

    def transplant(self, depriciatedTree, newTree):
        node = self.iterGenerator(depriciatedTree)
        if not isinstance(newTree, self.leaf) and isinstance(newTree, int):
            newTree = self.leaf(value=newTree)
        elif newTree is None or isinstance(newTree, Node):
            pass
        elif not isinstance(newTree, int):
            raise TypeError(f"{type(newTree)} is not supported yet.")
        if node is self.root:
            self.root = newTree
        elif node.parent.right is node:
            node.parent.right = newTree
        else:
            node.parent.left = newTree
        try:
            newTree.parent = node.parent
        finally:
            return node

    def remove(self, toRemove):
        toDelete = self.iterGenerator(toRemove)
        if toDelete.repititions:
            temp = toDelete.copy()
            toDelete.repititions -= 1
            return temp
        if Node.is_sentinel(toDelete.left):
            return self.transplant(toDelete, toDelete.right)
        elif Node.is_sentinel(toDelete.right):
            return self.transplant(toDelete, toDelete.left)
        else:
            replacement = self.next(toDelete, returnNode=True)
            self.transplant(replacement, replacement.right)
            replacement.right, replacement.left = toDelete.right, toDelete.left
            return self.transplant(toDelete, replacement)

    def rotate(self, parentLeaf, left=False, right=False):
        # Parent based rotation
        if (left and right) or (not left and not right):
            raise Exception("Rotation direction unclear")
        elif left:
            if Node.is_sentinel(parentLeaf.right):
                raise Exception("Type None rotations are not supported")
            rightChild, parentLeaf.right = parentLeaf.right, parentLeaf.right.left
            if not Node.is_sentinel(rightChild.left):
                rightChild.left.parent = parentLeaf
            rightChild.parent = parentLeaf.parent
            if Node.is_sentinel(parentLeaf.parent):
                self.root = rightChild
            elif parentLeaf.parent.right is parentLeaf:
                parentLeaf.parent.right = rightChild
            else:
                parentLeaf.parent.left = rightChild
            rightChild.left, parentLeaf.parent = parentLeaf, rightChild
        elif right:
            if Node.is_sentinel(parentLeaf.left):
                raise Exception("Type None rotations are not supported")
            leftChild, parentLeaf.left = parentLeaf.left, parentLeaf.left.right
            if not Node.is_sentinel(leftChild.right):
                leftChild.right.parent = parentLeaf
            leftChild.parent = parentLeaf.parent
            if Node.is_sentinel(parentLeaf.right):
                self.root = leftChild
            elif parentLeaf.parent.left is parentLeaf:
                parentLeaf.parent.left = leftChild
            else:
                parentLeaf.parent.right = leftChild
            leftChild.right, parentLeaf.parent = parentLeaf, leftChild

    def inorder(self):
        sortedList = list()

        def treeWalk(node: BinarySearchTree.leaf):
            if node is not None and not Node.is_sentinel(node):
                nonlocal sortedList
                treeWalk(node.left)
                sortedList.append(node.value)
                treeWalk(node.right)

        treeWalk(self.root)
        return sortedList

    @staticmethod
    def height(tree) -> int:
        if tree is None:
            return 1
        return max(BinarySearchTree.height(tree.left), BinarySearchTree.height(tree.right)) + 1


class RedBlackTree(BinarySearchTree):

    class leaf(Node):
        __slots__ = ['parent', 'right', 'left',
                     'value', 'repititions', 'colour']

        def __init__(self, parent=None, rightChild=None, leftChild=None, value=int(), repititions=int(), colour=RED):
            super().__init__(parent=parent, rightChild=rightChild,
                             leftChild=leftChild, value=value, repititions=repititions)
            self.colour = colour

        def __eq__(self, other):
            if super().__eq__(other):
                return self.colour == other.colour
            else:
                return False

        def __str__(self):
            return super().__str__() + f"colour: {'Black' if self.colour == BLACK else 'Red'}"

    sentinel = leaf(parent=None, value=None, colour=BLACK,
                    rightChild=None, leftChild=None)

    def __init__(self, *args):
        elements = RedBlackTree._breaker(args)
        self.root = self.leaf(value=elements[0], parent=RedBlackTree.sentinel,
                              rightChild=RedBlackTree.sentinel, leftChild=RedBlackTree.sentinel, colour=BLACK)
        del elements[0]
        self.insert(elements)

    def insert(self, *toAdd):
        newElements = RedBlackTree._breaker(toAdd)
        for num in newElements:
            iterator = self.root
            while True:
                if num == iterator.value:
                    iterator.repititions += 1
                    break
                elif num > iterator.value:
                    if iterator.right != RedBlackTree.sentinel:
                        iterator = iterator.right
                    else:
                        iterator.right = self.leaf(
                            parent=iterator, value=num, rightChild=RedBlackTree.sentinel, leftChild=RedBlackTree.sentinel)
                        self.insertionFixup(iterator.right)
                        break
                else:
                    if iterator.left != RedBlackTree.sentinel:
                        iterator = iterator.left
                    else:
                        iterator.left = self.leaf(
                            parent=iterator, value=num, rightChild=RedBlackTree.sentinel, leftChild=RedBlackTree.sentinel)
                        self.insertionFixup(iterator.left)
                        break

    def insertionFixup(self, AddedElement):
        # They are 3 cases in general,
        # Case 1: Inserted leaf's uncle is red then make the father and the uncle black and
        # make inserted node's grandfather red and shift index form inserted node to grandfather (the now trouble causing node)
        # Case 2: Inserted Leaf's uncle is black and it's parent is a right child and the inserted node itself is a left child(forming a triangle),
        # perform a left rotation on this node to make it a left child (similarly if it's parent left child and the inserted node a right we perform a right rotation)
        # Case 3: inserted leaf's parent is a right child and inserted leaf is a right child
        # Basically the counterpart of Case 2, child oriented just like its parent. parent is right and so is its kid

        currLeaf = self.iterGenerator(AddedElement)
        while currLeaf.parent.colour is RED:
            if currLeaf.parent is currLeaf.parent.parent.left:
                uncle_currLeaf = currLeaf.parent.parent.right
                if uncle_currLeaf is RED:
                    uncle_currLeaf = currLeaf.parent = BLACK
                    currLeaf.parent.parent.colour, currLeaf = RED, currLeaf.parent.parent  # case 1
                else:
                    if currLeaf.parent.right is currLeaf:
                        currLeaf = currLeaf.parent  # case 2
                        self.rotate(currLeaf, left=True)
                    currLeaf.parent.colour, currLeaf.parent.parent.colour = BLACK, RED  # case 3
                    self.rotate(currLeaf.parent.parent, right=True)
            else:
                uncle_currLeaf = currLeaf.parent.parent.left
                if uncle_currLeaf is RED:
                    uncle_currLeaf = currLeaf.parent = BLACK
                    currLeaf.parent.parent.colour, currLeaf = RED, currLeaf.parent.parent               # case 1
                else:
                    if currLeaf.parent.left is currLeaf:
                        # case 2
                        currLeaf = currLeaf.parent
                        self.rotate(currLeaf, right=True)
                    currLeaf.parent.colour, currLeaf.parent.parent.colour = BLACK, RED                  # case 3
                    self.rotate(currLeaf.parent.parent, left=True)
        self.root.colour = BLACK

    def remove(self, toRemove):
        toDelete = self.iterGenerator(toRemove)
        if toDelete.repititions:
            temp = toDelete.copy()
            toDelete.repititions -= 1
            return temp
        suspected_discrepant_leaf, ogColour = self.leaf(), toDelete.colour
        if toDelete.left == RedBlackTree.sentinel:
            suspected_discrepant_leaf = toDelete.right
            self.transplant(toDelete, toDelete.right)
        elif toDelete.right == RedBlackTree.sentinel:
            suspected_discrepant_leaf = toDelete.left
            self.transplant(toDelete, toDelete.left)
        else:
            replacement = self.next(toDelete, returnNode=True)
            ogColour, suspected_discrepant_leaf = replacement.colour, replacement.right
            if replacement.parent is toDelete:
                suspected_discrepant_leaf.parent = replacement  # there might be a sentinel here
            else:
                self.transplant(replacement, replacement.right)
                replacement.right = toDelete.right
                replacement.right.parent = replacement
            self.transplant(toDelete, replacement)
            replacement.left, replacement.colour = toDelete.left, toDelete.colour
            replacement.left.parent = replacement
        if ogColour is BLACK:
            self.deletionFixup(discrepantLeaf=suspected_discrepant_leaf)
        return toDelete

    def deletionFixup(self, discrepantLeaf):
        # They are 4 cases for deletion fixup
        # Since inserted node is red, we always look at the sibling of the inserted node
        # Case 1: If sibling is red then make sibling black and make inserted leaf's parent red
        #         rotate the sibling such that the parent rotates towards the inserted node.
        #         (ie. if inserted was a left child then the parent is left rotated)
        #         this changes inserted node's sibling.
        # Case 2: Inserted Node's sibling's kids are both black
        #         (irrespective sibling's colour, it might seem confusing but for now don't think about what we just did in Case 1)
        #         make sibling's colour red & colour inserted node's colour of the same colour as its parent.
        # Case 3: Inserted Node's sibling's far child is black
        #         (if Inserted node is a left child, then the far child would its sibling's right child)
        #         since either this case (& case 4) or case 2 would be performed we can assume atleast one of inserted node's
        #         sibling's kid is red. Purpose of this case is just to make inserted node's sibling's far child red.
        #         Colour inserted node's sibling's non far child black & the inserted node's sibling red.
        #         Perform a rotation away from the inserted node on inserted node's sibling. Now inserted node's sibling's far child is red.
        # Case 4: Inserted Node's sibling's far child is red
        #         (if inserted Node's sibling's far child is red then we jump directly to case 4)
        #         Colour inserted node's sibling's colour as same its parent
        #         Colour inserted node's sibling's far child & inserted node's parent black
        #         perform a rotation on inserted node's parent towards inserted node.
        #         (if inserted node is a right child then perform a right rotate)

        while discrepantLeaf is not self.root and discrepantLeaf.colour is BLACK:
            if discrepantLeaf.parent.left is discrepantLeaf:
                sibling_discrepantLeaf = discrepantLeaf.parent.right
                if sibling_discrepantLeaf.colour is RED:
                    sibling_discrepantLeaf.colour, discrepantLeaf.parent.colour = BLACK, RED
                    self.rotate(discrepantLeaf.parent, left=True)
                    sibling_discrepantLeaf = discrepantLeaf.parent.right
                if sibling_discrepantLeaf.right.colour == sibling_discrepantLeaf.left.colour == BLACK:
                    sibling_discrepantLeaf.colour, discrepantLeaf = RED, discrepantLeaf.parent
                else:
                    if sibling_discrepantLeaf.right.colour is BLACK:
                        sibling_discrepantLeaf.left.colour, sibling_discrepantLeaf.colour = BLACK, RED
                        self.rotate(sibling_discrepantLeaf, right=True)
                        sibling_discrepantLeaf = discrepantLeaf.parent.right
                    sibling_discrepantLeaf.colour = discrepantLeaf.parent.colour
                    sibling_discrepantLeaf.right.colour = discrepantLeaf.parent.colour = BLACK
                    self.rotate(discrepantLeaf.parent, left=True)
                    break
            else:
                sibling_discrepantLeaf = discrepantLeaf.parent.left
                if sibling_discrepantLeaf is RED:
                    sibling_discrepantLeaf.colour, discrepantLeaf.parent.colour = BLACK, RED
                    self.rotate(discrepantLeaf.parent, right=True)
                    sibling_discrepantLeaf = discrepantLeaf.parent.left
                if sibling_discrepantLeaf.right == sibling_discrepantLeaf.left == BLACK:
                    sibling_discrepantLeaf.colour, discrepantLeaf = RED, discrepantLeaf.parent
                else:
                    if sibling_discrepantLeaf.left is BLACK:
                        sibling_discrepantLeaf.right.colour, sibling_discrepantLeaf.colour = BLACK, RED
                        self.rotate(sibling_discrepantLeaf, left=True)
                    sibling_discrepantLeaf.colour = sibling_discrepantLeaf.parent.colour
                    sibling_discrepantLeaf.left.colour = discrepantLeaf.parent.colour = BLACK
                    self.rotate(discrepantLeaf.parent, right=True)
                    break
        discrepantLeaf.colour = BLACK


class SplayTree(BinarySearchTree):

    class leaf(Node):
        def __init__(self, parent=None, rightChild=None, leftChild=None, value=int(), repititions=int()):
            super().__init__(parent, rightChild, leftChild, value, repititions)

    sentinel = leaf()

    def __init__(self, *args):
        elements = SplayTree._breaker(args)
        rootVal = nearMatching(elements, mean(elements))
        self.root = self.leaf(value=elements[rootVal], parent=SplayTree.sentinel,
                              rightChild=SplayTree.sentinel, leftChild=SplayTree.sentinel)
        del elements[rootVal]
        self.insert(elements)

    def __call__(self):
        return NotImplemented

    # zig means you need to rotate to the right
    def zig(self, node: leaf):
        if node.parent is self.root and node.parent.left is node:
            return True
        return False

    # zag means you need to rotate to the left
    def zag(self, node: leaf):
        if node.parent is self.root and node.parent.right is node:
            return True
        return False

    # zig-zig means it's a left chain, so you rotate right twice
    def zig_zig(self, node: leaf):
        if node.parent is not self.root and node.parent.left is node and node.parent.parent.left is node.parent:
            return True
        return False

    # zag-zag means it's a right chain, so you rotate left twice
    def zag_zag(self, node: leaf):
        if node.parent is not self.root and node.parent.right is node and node.parent.parent.right is node.parent:
            return True
        return False

    # zig-zag means you do zig (rotate to the right) then zag (rotate to the left). it's right child then left child chain
    def zig_zag(self, node: leaf):
        if node.parent is not self.root and node.parent.left is node and node.parent.parent.right is node.parent:
            return True
        return False

    # zag-zig means you do zag (rotate to the left) then zig (rotate to the right). it's left child then right child chain
    def zag_zig(self, node: leaf):
        if node.parent is not self.root and node.parent.right is node and node.parent.parent.left is node.parent:
            return True
        return False

    def splay(self, value):
        nodeToSplay = self.iterGenerator(value)
        while nodeToSplay is not self.root:
            if self.zig(nodeToSplay):
                self.rotate(nodeToSplay.parent, right=True)

            elif self.zag(nodeToSplay):
                self.rotate(nodeToSplay.parent, left=True)

            elif self.zig_zig(nodeToSplay):
                self.rotate(nodeToSplay.parent.parent, right=True)
                self.rotate(nodeToSplay.parent, right=True)

            elif self.zag_zag(nodeToSplay):
                self.rotate(nodeToSplay.parent.parent, left=True)
                self.rotate(nodeToSplay.parent, left=True)

            elif self.zig_zag(nodeToSplay):
                self.rotate(nodeToSplay.parent, right=True)
                self.rotate(nodeToSplay.parent, left=True)

            elif self.zag_zig(nodeToSplay):
                self.rotate(nodeToSplay.parent, left=True)
                self.rotate(nodeToSplay.parent, right=True)

        return self.root

    def greatestMin(self, value, returnNode=True):
        itr = self.root
        while True:
            if itr.value < value and itr.right is not SplayTree.sentinel and self.min(itr.right) < value:
                itr = itr.right
                continue
            elif itr.value > value and itr.left is not SplayTree.sentinel:
                itr = itr.left
                continue
            elif returnNode:
                return itr
            else:
                return itr.value

    def smallestMax(self, value, returnNode=True):
        itr = self.root
        while True:
            if itr.value < value and itr.right is not SplayTree.sentinel:
                itr = itr.right
                continue
            elif itr.value > value and itr.left is not SplayTree.sentinel and self.max(itr.left) > value:
                itr = itr.left
                continue
            if returnNode:
                return itr
            return itr.value

    def insert(self, *toAdd):
        newElements = SplayTree._breaker(toAdd)
        for newElement in newElements:
            print(self.root)
            rightHeight, leftHeight, toSplay = SplayTree.height(
                self.root.right), SplayTree.height(self.root.left), self.leaf

            if rightHeight > leftHeight:
                toSplay = self.greatestMin(newElement)
            elif leftHeight > rightHeight:
                toSplay = self.smallestMax(newElement)
            else:
                toSplay = self.greatestMin(newElement) if random.choice(
                    [False, True]) else self.smallestMax(newElement)

            if toSplay.value != newElement:
                splayed, newRoot = self.splay(toSplay), self.leaf()
                if toSplay.value > newElement:
                    newRoot = self.leaf(
                        parent=SplayTree.sentinel, rightChild=splayed, leftChild=splayed.left, value=newElement)
                    splayed.left.parent, splayed.left = newRoot, SplayTree.sentinel
                else:
                    newRoot = self.leaf(
                        parent=SplayTree.sentinel, rightChild=splayed.right, leftChild=splayed, value=newElement)
                    splayed.right.parent, splayed.right = newRoot, SplayTree.sentinel
                splayed.parent = self.root = newRoot

    def delete(self, toDelete):
        nodeToDelete, newRoot, rightHeight, leftHeight, chc = self.splay(
            toDelete), self.leaf, SplayTree.height(self.root.right), SplayTree.height(self.root.left), None
        if rightHeight > leftHeight:
            chc = True
            newRoot = self.next(nodeToDelete, returnNode=True)
        elif rightHeight < leftHeight:
            chc = False
            newRoot = self.prev(nodeToDelete, returnNode=True)
        else:
            chc = random.choice([False, True])
            newRoot = self.next(nodeToDelete, returnNode=True) if chc else self.prev(
                nodeToDelete, returnNode=True)

        if chc:
            if self.root.right is newRoot:
                newRoot.left, self.root.left.parent, newRoot.parent = self.root.left, newRoot, SplayTree.sentinel
            else:
                if newRoot.right is not SplayTree.sentinel:
                    newRoot.right.parent = newRoot.parent
                newRoot.parent.left = newRoot.right
                newRoot.left, newRoot.right, newRoot.parent = self.root.left, self.root.right, self.root.parent
                if self.root.left is not SplayTree.sentinel:
                    self.root.left.parent = newRoot
        else:
            if self.root.left is newRoot:
                newRoot.right, self.root.right.parent, newRoot.parent = self.root.right, newRoot, SplayTree.sentinel
            else:
                if newRoot.left is not SplayTree.sentinel:
                    newRoot.left.parent = newRoot.parent
                newRoot.parent.right = newRoot.left
                newRoot.left, newRoot.right, newRoot.parent = self.root.left, self.root.right, self.root.parent
                if self.root.right is not SplayTree.sentinel:
                    self.root.right.parent = newRoot
        self.root = newRoot
        del nodeToDelete


class Treap(BinarySearchTree):
    class leaf(Node):
        __slots__ = ['parent', 'right', 'left',
                     'value', 'repititions', 'priority']

        def __init__(self, parent=None, rightChild=None, leftChild=None, value=int(), repititions=int(), priority=int()):
            super().__init__(parent=parent, rightChild=rightChild,
                             leftChild=leftChild, value=value, repititions=repititions)
            self.priority = priority

        def __eq__(self, other):
            if super().__eq__(other):
                return self.priority == other.priority
            else:
                return False

        def __str__(self):
            return super().__str__() + f"priority: {self.priority}"

    sentinel = leaf()

    def __init__(self, values: Iterable, priorities: Iterable):
        if priorities is None:
            priorities, values = tuple(random.sample(
                range(len(values)**2), k=len(values))), tuple(values)
        elements = Treap.make_pair(values, priorities)
        if isinstance(elements, zip):
            for index, pair in enumerate(elements):
                if index == 0:
                    self.root = self.leaf(
                        value=pair[0], parent=Treap.sentinel, rightChild=Treap.sentinel, leftChild=Treap.sentinel, priority=pair[1])
                else:
                    self.insert(pair)
        del elements

    def insert(self, element: tuple):
        iterator = self.root
        while True:
            if element[0] == iterator.value and element[1] == iterator.priority:
                iterator.repititions += 1
                break
            elif element[0] >= iterator.value:
                if iterator.right != Treap.sentinel:
                    iterator = iterator.right
                else:
                    iterator.right = self.leaf(
                        parent=iterator, value=element[0], rightChild=Treap.sentinel, leftChild=Treap.sentinel, priority=element[1])
                    self.insertionFixup(iterator.right)
                    break
            else:
                if iterator.left != RedBlackTree.sentinel:
                    iterator = iterator.left
                else:
                    iterator.left = self.leaf(
                        parent=iterator, value=element[0], rightChild=Treap.sentinel, leftChild=Treap.sentinel, priority=element[1])
                    self.insertionFixup(iterator.left)
                    break

    def insertionFixup(self, problematicNode: leaf):
        while problematicNode.parent is not Treap.sentinel and problematicNode.priority < problematicNode.parent.priority:
            if problematicNode.parent.right is problematicNode:
                self.rotate(problematicNode.parent, left=True)
            else:
                self.rotate(problematicNode.parent, right=True)

    def delete(self, toDelete):
        toDelete = self.remove(toDelete)
        self.deletionFixup(toDelete)

    def deletionFixup(self, toFix):
        minPriority = min(toFix.priority, toFix.left.priority if toFix.left is not Treap.sentinel else float(
            "-inf"), toFix.right.priority if toFix.right is not Treap.sentinel else float("-inf"))
        if minPriority is not toFix.priority:
            if toFix.left is Treap.sentinel or (toFix.right is not Treap.sentinel and toFix.left.priority > toFix.right.priority):
                self.rotate(toFix, left=True)
            else:
                self.rotate(toFix, right=True)
            self.deletionFixup(toFix)

    @staticmethod
    def make_pair(value, priority):
        if type(value) != type(priority):
            raise TypeError(
                f"Data type of parameters should be same. Received {type(value).__name__} and {type(priority).__name__}")
        if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
            return zip(Treap._breaker(value), Treap._breaker(priority))
        elif isinstance(priority, int):
            return tuple(value, priority)
        else:
            raise TypeError(
                f"{type(value)} or {type(priority)} not supported.")


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
