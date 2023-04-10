from myFunctions import execute_this, nearMatching
from random import randint as rint
from math import ceil
from statistics import mean
from collections import Counter


def bubbleSort_itr(UnsortedList):

    end = len(UnsortedList)
    for x in range(end):
        for y in range(x + 1, end):
            if UnsortedList[x] > UnsortedList[y]:
                UnsortedList[y], UnsortedList[x] = UnsortedList[x], UnsortedList[y]

    return (UnsortedList)


def insertionSort(UnsortedList):
    # (swap at every iteration till it goes itll its desired position(in a particular round)), faster than merge sort for small lists

    for x in range(1, len(UnsortedList)):
        y = x
        while y > 0 and UnsortedList[y - 1] > UnsortedList[y]:
            UnsortedList[y-1], UnsortedList[y] = UnsortedList[y], UnsortedList[y-1]
            y -= 1

    return (UnsortedList)


def insertionSort_recur(UnsortedList):

    if len(UnsortedList) > 1:
        nth_ele = UnsortedList.pop()
        UnsortedList = insertionSort_recur(UnsortedList)

        for x in range(len(UnsortedList)):
            if nth_ele < UnsortedList[x]:
                UnsortedList.insert(x, nth_ele)
                break
        else:
            UnsortedList.append(nth_ele)

    return UnsortedList


def mergeSort(UnsortedList):
    if len(UnsortedList) > 1:
        mid = len(UnsortedList) // 2
        left, right = mergeSort(UnsortedList[:mid]), mergeSort(UnsortedList[mid:])
        UnsortedList = list()

        while len(left) or len(right):
            try:
                assert left[0] is not None and right[0] is not None
            except:
                if len(right):
                    UnsortedList.extend(right)
                    right.clear()
                else:
                    UnsortedList.extend(left)
                    left.clear()
            else:
                if left[0] < right[0]:
                    UnsortedList.append(right[0])
                    right.pop(0)
                else:
                    UnsortedList.append(left[0])
                    left.pop(0)

    return (UnsortedList)


def selectionSort(UnsortedList):

    for x in range(len(UnsortedList)-1):
        min_key = x

        for y in range(x+1, len(UnsortedList)):
            if UnsortedList[y] < UnsortedList[min_key]:
                min_key = y

        UnsortedList[x], UnsortedList[min_key] = UnsortedList[min_key], UnsortedList[x]

    return (UnsortedList)


def heapSort(UnsortedList):

    def Child(index):
        for n in range(1, 2+1):
            yield (2*index+n)

    def maxHeap(List, index):
        kids, largest = Child(index), index
        for kid in kids:
            if kid >= len(List):
                break
            if List[kid] > List[largest]:
                largest = kid
        if largest != index:
            List[index], List[largest] = List[largest], List[index]
            maxHeap(List, largest)
        return List

    def Heapify(List):
        if len(List) == 1:
            return (List)
        for index in range(len(List) // 2, -1, -1):
            maxHeap(List, index)
        return (List)

    def Sorting(UnsortedList):
        sortedList = list()
        while UnsortedList:
            UnsortedList = Heapify(UnsortedList)
            sortedList.insert(0, UnsortedList[0])
            del UnsortedList[0]
        return (sortedList)

    return Sorting(UnsortedList)


def quickSort(UnsortedList):

    def myPartition(UnsortedList, lowEnd, highEnd):
        tempList = UnsortedList[lowEnd:highEnd]
        leftPartition, rightPartition, pivot = lowEnd - 1, lowEnd, UnsortedList[nearMatching(tuple(tempList), mean(tempList))+lowEnd]
        for x in range(lowEnd, highEnd):
            # changing the sign to greater than will sort in descending order
            if UnsortedList[x] < pivot:
                UnsortedList[rightPartition], UnsortedList[x] = UnsortedList[x], UnsortedList[rightPartition]
                UnsortedList[leftPartition + 1], UnsortedList[rightPartition] = UnsortedList[rightPartition], UnsortedList[leftPartition+1]
                leftPartition, rightPartition = leftPartition+1, rightPartition + 1
            elif UnsortedList[x] == pivot:
                UnsortedList[rightPartition], UnsortedList[x] = UnsortedList[x], UnsortedList[rightPartition]
                rightPartition += 1
        return (leftPartition+1, rightPartition)

    def HoarePartition(UnsortedList, lowEnd, highEnd):
        # Hoare implemented a multi index algorithm one starts from left and the other from the right.
        # OG partition technique although not used here
        leftIndex, rightIndex = lowEnd, highEnd-1
        pivot = UnsortedList[lowEnd]
        while True:
            while rightIndex >= lowEnd and UnsortedList[rightIndex] > pivot:
                rightIndex -= 1
            while leftIndex < highEnd and UnsortedList[leftIndex] < pivot:
                leftIndex += 1
            if leftIndex < rightIndex:
                UnsortedList[leftIndex], UnsortedList[rightIndex] = UnsortedList[rightIndex], UnsortedList[leftIndex]
            else:
                return (rightIndex)

    def divide(UnsortedList, lowEnd, highEnd):
        if lowEnd < highEnd:
            lower, mid = myPartition(UnsortedList, lowEnd, highEnd)
            divide(UnsortedList, lowEnd, lower)
            divide(UnsortedList, mid, highEnd)
        return

    divide(UnsortedList, 0, len(UnsortedList))
    return(UnsortedList)


def countSort(UnsortedList):
    # used for small inputs due to memory requirements
    UnsortedList = tuple(UnsortedList)
    countedElements, maxEle, minEle = Counter(UnsortedList), max(UnsortedList)+1, min(UnsortedList)
    tempList, SortedList = [countedElements[x]for x in range(minEle, maxEle)], [0]*len(UnsortedList)
    del countedElements

    for x in range(1, maxEle-minEle):
        tempList[x] += tempList[x - 1]

    for x in range(len(UnsortedList)-1, -1, -1):
        SortedList[tempList[UnsortedList[x]-minEle]-1] = UnsortedList[x]
        tempList[UnsortedList[x]-minEle] -= 1

    return (SortedList)


def bucketSort(UnsortedList):
    # Bucket can be of any type. Here bucket is made based on range of numbers but they can be made based on length on objects or type of objects,
    # for eg. 2 digit object, 3 digit object, Making bucket of each letter in the alphabet
    UnsortedList, Buckets = tuple(UnsortedList), dict()
    minEle, totalBuckets = min(UnsortedList), ceil(len(UnsortedList)/2)
    maxBucketSize = ceil((max(UnsortedList)-minEle) / totalBuckets)

    for x in range(totalBuckets):
        Buckets[x] = list()

    for element in UnsortedList:
        try:
            Buckets[(element - minEle) // maxBucketSize].append(element)
        except KeyError:
            Buckets[totalBuckets-1].append(element)

    for x in range(totalBuckets):
        Buckets[x].sort()

    dummy = list()
    for x in range(totalBuckets):
        dummy.extend(Buckets[x])
    return dummy


def shellSort(UnsortedList):
    gap = len(UnsortedList) // 2
    while gap:
        for x in range(gap, len(UnsortedList), gap):
            temp, j = UnsortedList[x], x
            while j >= gap and UnsortedList[j - gap] > temp:
                UnsortedList[j] = UnsortedList[j - gap]
                j -= gap
            UnsortedList[j] = temp
        gap //= 2
    return UnsortedList

@execute_this
def sorting():
    a = [x for x in range(10)]
    return mergeSort(a)

print(sorting)
