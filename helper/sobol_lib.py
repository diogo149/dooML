import numpy as np
import math


def high_one_bit(n):
    """ returns the position of the high 1 bit base 2 in an integer or 0 if less than or equal to 1
    """
    if n <= 0:
        return 0
    return int(math.log(n) / math.log(2)) + 1


def low_zero_bit(n):
    """ returns the position of the low 0 bit base 2 in an integer or 0 if less than or equal to 1
    """
    if n <= 0:
        return 1
    bit = 1
    while n & 1:
        bit += 1
        n >>= 1
    return bit


def i4_uniform(a, b, seed):
    """ returns a scaled pseudorandom I4.
    """
    a, b = int(min(a, b)), int(max(a, b))
    seed = int(seed) % 2147483647
    k = seed // 127773
    seed = (16807 * (seed - k * 127773) - k * 2836) % 2147483647
    r = seed * 4.656612875E-10
    r = (1.0 - r) * a + r * (b + 1)
    return [int(r), seed]


def i4_sobol_generate(m, n, skip):
    """ generates a Sobol dataset
    """
    if m > 40:
        return np.random.uniform(size=(m, n))
    r = np.zeros((m, n))
    for j in xrange(1, n + 1):
        seed = skip + j - 2
        [r[0:m, j - 1], seed] = i4_sobol(m, seed)
    return r


def i4_sobol(dim_num, seed):
    """generates a new quasirandom Sobol vector with each call
    """
    # constants
    DIM_MAX = 40
    MAX_COL = 30
    seed = int(max(seed, 0))
    col = low_zero_bit(seed)
    assert col <= MAX_COL
    assert 1 <= dim_num <= DIM_MAX, dim_num

    POLY = [1, 3, 7, 11, 13, 19, 25, 37, 59, 47,
            61, 55, 41, 67, 97, 91, 109, 103, 115, 131,
            193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
            213, 191, 253, 203, 211, 239, 247, 285, 369, 299]

    M = [high_one_bit(j) - 1 for j in POLY]

    V = np.zeros((DIM_MAX, MAX_COL), dtype=np.int)
    V[0:40, 0] = [1] * 40
    V[2:40, 1] = [1, 3] * 19
    V[3:40, 2] = [7, 5, 1, 3, 3, 7, 5,
                  5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
                  5, 3, 3, 1, 7, 5, 1, 3, 3, 7,
                  5, 1, 1, 5, 7, 7, 5, 1, 3, 3]
    V[5:40, 3] = [1, 7, 9, 13, 11,
                  1, 3, 7, 9, 5, 13, 13, 11, 3, 15,
                  5, 3, 15, 7, 9, 13, 9, 1, 11, 7,
                  5, 15, 1, 15, 11, 5, 3, 1, 7, 9]
    V[7:40, 4] = [9, 3, 27,
                  15, 29, 21, 23, 19, 11, 25, 7, 13, 17,
                  1, 25, 29, 3, 31, 11, 5, 23, 27, 19,
                  21, 5, 1, 17, 13, 7, 15, 9, 31, 9]
    V[13:40, 5] = [37, 33, 7, 5, 11, 39, 63,
                   27, 17, 15, 23, 29, 3, 21, 13, 31, 25,
                   9, 49, 33, 19, 29, 11, 19, 27, 15, 25]
    V[19:40, 6] = [13,
                   33, 115, 41, 79, 17, 29, 119, 75, 73, 105,
                   7, 59, 65, 21, 3, 113, 61, 89, 45, 107]
    V[37:40, 7] = [7, 23, 39]
    V[0, 0:MAX_COL] = 1

    for i in xrange(2, dim_num + 1):
        m = M[i - 1]
        includ = np.array(map(float, bin(POLY[i - 1])[3:]))

        for j in xrange(m + 1, MAX_COL + 1):
            new_v = V[i - 1, j - m - 1]
            l = 1
            for k in xrange(1, m + 1):
                l = 2 * l
                if (includ[k - 1]):
                    new_v = np.bitwise_xor(int(new_v), int(l * V[i - 1, j - k - 1]))
            V[i - 1, j - 1] = new_v

    for i, j in enumerate(xrange(MAX_COL - 1, 0, -1)):
        V[0:dim_num, j - 1] = V[0:dim_num, j - 1] * 2 ** (i + 1)

    lastq = np.zeros(dim_num, dtype=np.int)
    for seed_temp in xrange(0, seed):
        l_tmp = low_zero_bit(seed_temp)
        for i in xrange(1, dim_num + 1):
            lastq[i - 1] = np.bitwise_xor(lastq[i - 1], V[i - 1, l_tmp - 1])

    recipd = 1.0 / (2 ** MAX_COL)
    quasi = np.zeros(dim_num)
    for i in xrange(1, dim_num + 1):
        quasi[i - 1] = lastq[i - 1] * recipd
        lastq[i - 1] = np.bitwise_xor(lastq[i - 1], V[i - 1, col - 1])

    return (quasi, seed + 1)

if __name__ == "__main__":
    assert np.all(i4_sobol_generate(3, 4, 2) == np.array([[0.5, 0.75, 0.25, 0.375], [0.5, 0.25, 0.75, 0.375], [0.5, 0.75, 0.25, 0.625]])), i4_sobol_generate(3, 4, 2)
    print i4_sobol_generate(3, 4, 2)
    print i4_sobol_generate(4, 3, 3)
    print i4_sobol_generate(10, 10, 3)
