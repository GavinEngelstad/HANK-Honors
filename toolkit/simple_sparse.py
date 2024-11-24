'''
Simple Sparse matrix following Aulcert et al "Using the Sequence-Space Jacobian to Solve
and Estimate Heterogeneous-Agent Models."

Uses shifting to create a computationally efficeint version of a sparse matrix where the
entires follow the diagnal. Can be used as an ordinary linear operator (i.e. can be added,
subtracted, multiplied, and, importantly, matrix multiplied), This is used for the Jacobian
of non-HA blocks
'''

## load some packages
import numpy as np  # written atop basic matrix operations from numpy
import numba as nb  # make things fast


class SimpleSparse(object):
    __array_priority__ = 1000

    # this is mostly copied from aulclert code
    def __init__(self, iarr, jarr, xarr):
        # convert to numpy arrays
        iarr, jarr, xarr = np.array(iarr), np.array(jarr), np.array(xarr)

        # remove zeros
        nonzero = xarr != 0
        iarr, jarr, xarr = iarr[nonzero], jarr[nonzero], xarr[nonzero]

        # stop it from repeating i, j pairs
        (iarr, jarr), inv = np.unique(np.vstack((iarr, jarr)), axis=1, return_inverse=True)
        xarr_unique = np.zeros_like(iarr, dtype=float)
        for k, x in zip(inv, xarr):
            xarr_unique[k] += x

        # i, j, x should have same length and be 1d
        self.iarr = iarr  # shifts
        self.jarr = jarr  # zeros
        self.xarr = xarr_unique  # coefficients

    # matrix things
    @property
    def T(self):
        return SimpleSparse(-self.iarr, self.jarr, self.xarr)
    
    def to_numpy(self, T):
        return self + np.zeros((T, T))
    
    # equality
    def __eq__(self, s):
        return (self.iarr == s.iarr).all() and (self.jarr == s.jarr).all() and (self.xarr == s.xarr).all()

    # positive and neagtive versions
    def __pos__(self):
        return self
    
    def __neg__(self):
        return SimpleSparse(self.iarr, self.jarr, -self.xarr)
    
    # scalar multiplication things
    def __mul__(self, a):
        return SimpleSparse(self.iarr, self.jarr, a * self.xarr)

    def __rmul__(self, a):  # called when a * Sparse
        return self * a  # calls __mul__
    
    def __truediv__(self, a):
        return SimpleSparse(self.iarr, self.jarr, self.xarr / a)
    
    # addition things
    def __add__(self, A):
        if isinstance(A, SimpleSparse):  # simple sparse matrix, combine vectors, sum where theres overlap
            # combine arrays
            iarr = np.hstack((self.iarr, A.iarr))
            jarr = np.hstack((self.jarr, A.jarr))
            xarr = np.hstack((self.xarr, A.xarr))

            return SimpleSparse(iarr.ravel(), jarr.ravel(), xarr.ravel())
        
        ## assumed to be an array
        T = A.shape[0]

        # flatten, then we can increment by T+1 steps instead of having to index each row
        A = A.flatten()  # flatten makes copy, ravel changes original
        for i, j, x in zip(self.iarr, self.jarr, self.xarr):
            if i >= 0:
                A[T * i + (T + 1) * j::T + 1] += x
            else:
                A[-i + (T + 1) * j:(T + i) * T:T + 1] += x
        return A.reshape((T, T))

    def __radd__(self, A):
        return self + A  # call __add__
    
    def __sub__(self, A):
        return self + (-A)  # call __add

    def __rsub__(self, A):
        return -self + A  # call __add__ on inverse

    # matrix multiplcation things
    def __matmul__(self, A):
        if isinstance(A, SimpleSparse):
            # simple sparse mmut;liplcation rules, from auclert paper
            # i = i_1 + i_2
            iarr = self.iarr + A.iarr[:, None]  # use dimensions to pair up all possible options
            jarr = self.ss_ss_j(self.iarr, self.jarr, A.iarr[:, None], A.jarr[:, None], iarr)
            xarr = self.xarr * A.xarr[:, None]

            return SimpleSparse(iarr.ravel(), jarr.ravel(), xarr.ravel())
        
        ## assummed to be numpy array
        sA = np.zeros_like(A, dtype=float)  # simple sparse * A
        for i, j, x in zip(self.iarr, self.jarr, self.xarr):
            if i >= 0:
                sA += x * SimpleSparse.shift_i(SimpleSparse.zero_j(A, j), i)
            else:
                sA += x * SimpleSparse.zero_j(SimpleSparse.shift_i(A, i), j)
        return sA
    
    @staticmethod
    @nb.vectorize([nb.int64(nb.int64, nb.int64, nb.int64, nb.int64, nb.int64)])
    def ss_ss_j(i1, j1, i2, j2, isum):
        ## gets the j value for simple sparse multiplcation
        if i1 >= 0:
            if i2 >= 0:
                return max(j1 - i2, j2)
            else:  # i2 < 0
                return max(j1, j2) + min(i1, -i2)
        else:  # i1 < 0
            if i2 >= 0:
                if isum >= 0:
                    return max(j1 - i1 - i2, j2)
                else:  # isum < 0
                    return max(j2 + i1 + i2, j1)
            else:  # i2 < 0
                return max(j1, j2 + i1)

    @staticmethod
    def shift_i(A, i):
        # handle 0s
        if i == 0:
            return A.copy()

        # create shifted array
        S_A = np.zeros_like(A)
        if i > 0:
            S_A[i:] = A[:-i]
        else:
            S_A[:i] = A[-i:]

        return S_A

    @staticmethod
    def zero_j(A, j):
        # handle 0s
        Z_A = A.copy()
        if j == 0:
            return Z_A

        # create shifted array
        if j > 0:
            Z_A[:j] = 0
        else:
            Z_A[j:] = 0

        return Z_A

    def __rmatmul__(self, A):
        return (self.T @ A.T).T

    # stringify
    def __repr__(self):
        return f'SimpleSparse({', '.join([f'({i}, {j}): {x}' for i, j, x in zip(self.iarr, self.jarr, self.xarr)])})'
    
    # clone
    def copy(self):
        return SimpleSparse(self.iarr.copy(), self.jarr.copy(), self.xarr.copy())


if __name__ == '__main__':
    ## checks (need to add more)
    T = 10

    # check unit shift on vector
    for t in range(T):
        check = SimpleSparse(t * np.ones(1, dtype=int), np.zeros(1, dtype=int), np.ones(1)) @ np.arange(T)
        actual = np.hstack((np.zeros(t), np.arange(T - t)))
        assert (check == actual).all()
    for t in range(T):
        check = SimpleSparse(-t * np.ones(1, dtype=int), np.zeros(1, dtype=int), np.ones(1)) @ np.arange(T)
        actual = np.hstack((np.arange(t, T), np.zeros(t)))
        assert (check == actual).all()

    # check unti zeros on vector
    for t in range(T):
        check = SimpleSparse(np.zeros(1, dtype=int), t * np.ones(1, dtype=int), np.ones(1)) @ np.arange(T)
        actual = np.arange(T)
        actual[:t] = 0
        assert (check == actual).all()

    # check transpose
    for _ in range(10):
        N = 5
        iarr = np.random.choice(np.arange(2 * N + 1) - N, N)
        jarr = np.random.choice(np.arange(N + 1), N)
        xarr = np.random.uniform(0, 10, N)

        spmat = SimpleSparse(iarr, jarr, xarr)
        assert (spmat.T.to_numpy(T) == spmat.to_numpy(T).T).all()

    # check to numpy works
    for _ in range(10):
        N = 5
        iarr = np.random.choice(np.arange(2 * N + 1) - N, N)
        jarr = np.random.choice(np.arange(N + 1), N)
        xarr = np.random.uniform(0, 10, N)

        spmat = SimpleSparse(iarr, jarr, xarr)
        assert (spmat @ np.eye(T) == spmat.to_numpy(T)).all()

    # check scaler multiplcation works
    for a in range(10):
        N = 5
        iarr = np.random.choice(np.arange(2 * N + 1) - N, N)
        jarr = np.random.choice(np.arange(N + 1), N)
        xarr = np.random.uniform(0, 10, N)

        assert np.allclose((a * SimpleSparse(iarr, jarr, xarr)).to_numpy(T) - SimpleSparse(iarr, jarr, a * xarr).to_numpy(10), 0)
        assert np.allclose((SimpleSparse(iarr, jarr, xarr)).to_numpy(T) * a - SimpleSparse(iarr, jarr, a * xarr).to_numpy(10), 0)
