"""
Bag: a probabilistic priority queue integrated with a hashtable.
Core data structure for NARS resource allocation under AIKR.
"""

import math


class Bag:
    """
    A bag can contain items up to a constant capacity.
    Each item has a unique key and a priority in [0,1].

    Operations:
      put(item)    - add item, merge if same key exists, evict lowest if full
      get(key)     - remove and return item by key
      select()     - remove and return item with probability ~ priority
      put_back(item) - return item after processing (with decay)
    """

    def __init__(self, levels, bucket_capacity, label=""):
        self.levels = levels
        self.bucket_capacity = bucket_capacity
        self.capacity = levels * bucket_capacity
        self.label = label

        # Array of buckets (lists acting as queues)
        self.buckets = [[] for _ in range(levels)]
        # Hashtable: key -> (item, level_index)
        self.index = {}
        # Distributor for deterministic probabilistic selection
        self.distributor = self._make_distributor(levels)
        self.dist_pointer = 0
        self.size = 0

    def _make_distributor(self, L):
        """
        Build distributor array D of size L*(L+1)/2.
        D contains (i+1) copies of index i, then shuffled deterministically.
        """
        D = []
        for j in range(L):
            for _ in range(j + 1):
                D.append(j)
        # Deterministic shuffle using a simple LCG-style approach
        # (reproducible across runs for the same L)
        n = len(D)
        seed = 31
        for i in range(n - 1, 0, -1):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            j = seed % (i + 1)
            D[i], D[j] = D[j], D[i]
        return D

    def _priority_to_level(self, priority):
        """Map priority [0,1] to bucket level [0, levels-1]."""
        level = math.ceil(priority * self.levels) - 1
        return max(0, min(self.levels - 1, level))

    def _enter_level(self, item, level):
        """Insert item at given level, cascade overflow downward."""
        if level < 0:
            # Absolute forgetting: remove from index
            if item.key in self.index:
                del self.index[item.key]
                self.size -= 1
            return

        self.buckets[level].insert(0, item)
        self.index[item.key] = (item, level)

        if len(self.buckets[level]) > self.bucket_capacity:
            overflow = self.buckets[level].pop()
            self._enter_level(overflow, level - 1)

    def put(self, item):
        """Add item to bag; merge if key exists, evict lowest if full."""
        if item.key in self.index:
            old_item, old_level = self.index[item.key]
            # Remove old from bucket
            if old_item in self.buckets[old_level]:
                self.buckets[old_level].remove(old_item)
            del self.index[item.key]
            self.size -= 1
            # Merge
            item.merge(old_item)

        self.size += 1
        level = self._priority_to_level(item.priority)
        self._enter_level(item, level)

    def get(self, key):
        """Remove and return item by key, or None."""
        if key not in self.index:
            return None
        item, level = self.index[key]
        if item in self.buckets[level]:
            self.buckets[level].remove(item)
        del self.index[key]
        self.size -= 1
        return item

    def select(self):
        """Select an item with probability proportional to priority."""
        if self.size == 0:
            return None

        # Walk through distributor levels to find a non-empty bucket
        attempts = 0
        max_attempts = len(self.distributor)
        while attempts < max_attempts:
            self.dist_pointer = (self.dist_pointer + 1) % len(self.distributor)
            level = self.distributor[self.dist_pointer]
            if self.buckets[level]:
                item = self.buckets[level].pop(0)
                del self.index[item.key]
                self.size -= 1
                return item
            attempts += 1

        # Fallback: scan all levels top-down
        for level in range(self.levels - 1, -1, -1):
            if self.buckets[level]:
                item = self.buckets[level].pop(0)
                del self.index[item.key]
                self.size -= 1
                return item
        return None

    def put_back(self, item):
        """Return an item to the bag after processing (with priority decay)."""
        item.priority *= item.durability  # decay
        item.priority = max(0.001, item.priority)  # prevent zero
        self.size += 1
        level = self._priority_to_level(item.priority)
        self.buckets[level].insert(0, item)
        self.index[item.key] = (item, level)

    def peek(self, key):
        """Look up an item without removing it."""
        if key not in self.index:
            return None
        return self.index[key][0]

    def __len__(self):
        return self.size

    def __contains__(self, key):
        return key in self.index

    def all_items(self):
        """Return all items (for inspection)."""
        return [item for item, _ in self.index.values()]

    def __repr__(self):
        return f"Bag({self.label}, size={self.size}/{self.capacity})"
