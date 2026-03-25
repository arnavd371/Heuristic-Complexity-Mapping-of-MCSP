"""
Truth table representation with bitset optimization using numpy uint64 arrays.
"""
import numpy as np
import math
import random


class TruthTable:
    """
    Represents a Boolean function via its truth table.
    Internally stored as a numpy array of uint64 words (bitset).
    For n variables, there are 2^n entries.
    """

    _ONE = np.uint64(1)  # cached constant to avoid repeated conversions

    def __init__(self, n: int, table=None):
        self.n = n
        self.size = 1 << n  # 2^n
        self.num_words = max(1, math.ceil(self.size / 64))
        self._words = np.zeros(self.num_words, dtype=np.uint64)
        if table is not None:
            if len(table) != self.size:
                raise ValueError(f"Table length {len(table)} != 2^n = {self.size}")
            for i, v in enumerate(table):
                if v:
                    self.set_bit(i, 1)

    def get_bit(self, idx: int) -> int:
        word = idx >> 6  # idx // 64
        bit = idx & 63   # idx % 64
        return int((self._words[word] >> np.uint64(bit)) & self._ONE)

    def set_bit(self, idx: int, val: int):
        word = idx >> 6
        bit = idx & 63
        if val:
            self._words[word] |= self._ONE << np.uint64(bit)
        else:
            self._words[word] &= ~(self._ONE << np.uint64(bit))

    def to_array(self) -> np.ndarray:
        """Return a numpy array of 0/1 values of length 2^n."""
        result = np.zeros(self.size, dtype=np.uint8)
        for i in range(self.size):
            result[i] = self.get_bit(i)
        return result

    def to_list(self) -> list:
        return self.to_array().tolist()

    def __and__(self, other: 'TruthTable') -> 'TruthTable':
        result = TruthTable(self.n)
        result._words = self._words & other._words
        return result

    def __or__(self, other: 'TruthTable') -> 'TruthTable':
        result = TruthTable(self.n)
        result._words = self._words | other._words
        return result

    def __xor__(self, other: 'TruthTable') -> 'TruthTable':
        result = TruthTable(self.n)
        result._words = self._words ^ other._words
        return result

    def __invert__(self) -> 'TruthTable':
        result = TruthTable(self.n)
        result._words = ~self._words
        # Mask the last word to exactly 2^n bits
        remainder = self.size % 64
        if remainder != 0:
            mask = np.uint64((1 << remainder) - 1)
            result._words[-1] &= mask
        return result

    def popcount(self) -> int:
        """Count the number of set bits (Hamming weight)."""
        total = 0
        for w in self._words:
            v = int(w)
            while v:
                total += v & 1
                v >>= 1
        return total

    def hamming_weight(self) -> int:
        return self.popcount()

    def __eq__(self, other) -> bool:
        if not isinstance(other, TruthTable):
            return False
        if self.n != other.n:
            return False
        return np.array_equal(self._words, other._words)

    def __repr__(self) -> str:
        bits = ''.join(str(self.get_bit(i)) for i in range(self.size))
        return f"TruthTable(n={self.n}, table={bits})"

    @classmethod
    def from_string(cls, n: int, s: str) -> 'TruthTable':
        """Create TruthTable from a binary string of length 2^n."""
        tt = cls(n)
        if len(s) != (1 << n):
            raise ValueError(f"String length {len(s)} != 2^n = {1 << n}")
        for i, c in enumerate(s):
            tt.set_bit(i, int(c))
        return tt

    @classmethod
    def zero(cls, n: int) -> 'TruthTable':
        return cls(n)

    @classmethod
    def one(cls, n: int) -> 'TruthTable':
        tt = cls(n)
        for i in range(1 << n):
            tt.set_bit(i, 1)
        return tt

    @classmethod
    def variable(cls, n: int, var_idx: int) -> 'TruthTable':
        """Truth table for the var_idx-th input variable."""
        tt = cls(n)
        size = 1 << n
        for i in range(size):
            bit = (i >> var_idx) & 1
            tt.set_bit(i, bit)
        return tt

    @classmethod
    def parity(cls, n: int) -> 'TruthTable':
        """XOR of all n inputs."""
        tt = cls(n)
        size = 1 << n
        for i in range(size):
            p = bin(i).count('1') % 2
            tt.set_bit(i, p)
        return tt

    @classmethod
    def majority(cls, n: int) -> 'TruthTable':
        """Majority function: output 1 iff more than n/2 inputs are 1."""
        tt = cls(n)
        size = 1 << n
        threshold = n / 2
        for i in range(size):
            ones = bin(i).count('1')
            tt.set_bit(i, 1 if ones > threshold else 0)
        return tt

    @classmethod
    def random(cls, n: int, seed=None) -> 'TruthTable':
        rng = np.random.default_rng(seed)
        tt = cls(n)
        size = 1 << n
        num_words = max(1, math.ceil(size / 64))
        tt._words = rng.integers(0, np.iinfo(np.uint64).max, size=num_words, dtype=np.uint64)
        # Mask the last word
        remainder = size % 64
        if remainder != 0:
            mask = np.uint64((1 << remainder) - 1)
            tt._words[-1] &= mask
        return tt

    @classmethod
    def threshold(cls, n: int, k: int) -> 'TruthTable':
        """Threshold function: output 1 iff at least k inputs are 1."""
        tt = cls(n)
        size = 1 << n
        for i in range(size):
            ones = bin(i).count('1')
            tt.set_bit(i, 1 if ones >= k else 0)
        return tt
