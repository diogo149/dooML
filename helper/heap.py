# my implementation of a heap! one of my earlier python projects (:
import heapq


class Heap(object):

    """ python implementation of a min heap
    """

    # note: has_key has to be a function!
    def __init__(self, to_heap=None, key=None, mutable=False):
        if to_heap is None:
            to_heap = []
        self.heap = to_heap
        self.key = key
        self.mutable = mutable
        if key is not None:
            if mutable:
                self.heap = [[key(item), item] for item in to_heap]
            else:
                self.heap = [(key(item), item) for item in to_heap]
        heapq.heapify(self.heap)

    # note: this raises IndexError if heap is empty
    def pop(self):
        if self.key is None:
            return heapq.heappop(self.heap)
        else:
            return heapq.heappop(self.heap)[1]

    def push(self, item):
        if self.key is None:
            heapq.heappush(self.heap, item)
        elif self.mutable:
            heapq.heappush(self.heap, [self.key(item), item])
        else:
            heapq.heappush(self.heap, (self.key(item), item))

    def remove(self, index):
        temp = heapq.heappop(self.heap)
        if index > 0:
            self.heap[index - 1] = temp
            # note: _siftup(heap, index) is for making larger terms go towards the leafs
            heapq._siftdown(self.heap, 0, index - 1)

    # this function assumes that mutable is True and key exists, or heap was structured in a similar way
    def modify_key(self, index, new_key):
        old_key = self.heap[index][0]
        self.heap[index][0] = new_key
        if old_key > new_key:
            heapq._siftdown(self.heap, 0, index)
        elif old_key < new_key:
            heapq._siftup(self.heap, index)

    def index(self, item):
        if self.key is None:
            return self.heap.index(item)
        elif self.mutable:
            return self.heap.index([self.key(item), item])
        else:
            return self.heap.index((self.key(item), item))

    # replace item at index i with new item then keep heap property
    def replace(self, i, item):
        self.remove(i)
        self.push(item)

    def size(self):
        return len(self.heap)

    def __str__(self):
        return str(self.heap)

    def __repr__(self):
        return repr(self.heap)


class ConstSizeHeap(object):

    def __init__(self, max_size, key=None):
        self.max_size = max_size
        self.heap = Heap(key=key)

    def push(self, item):
        self.heap.push(item)
        while self.heap.size() > self.max_size:
            self.heap.pop()

    def pop(self):
        return self.heap.pop()

    def to_list(self):
        return list(self.heap.heap)

    def __str__(self):
        return str(self.heap)

    def __repr__(self):
        return repr(self.heap)
