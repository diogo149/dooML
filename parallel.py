"""Table of Contents
    -parmap
    -pipe_parmap
"""

import multiprocessing
from functools import partial


def parmap(func, in_vals, args=[], kwargs={}, n_jobs=1):
    """ easy parallel map, but it pickles input arguments and thus can't be used for dynamically generated functions
    """
    assert isinstance(n_jobs, int)
    assert n_jobs >= -1
    new_func = partial(func, *args, **kwargs)
    if n_jobs == 1:
        mapped = map(new_func, in_vals)
    else:
        if n_jobs == -1:
            pool = multiprocessing.Pool()
        else:
            pool = multiprocessing.Pool(processes=n_jobs)
        mapped = pool.map(new_func, in_vals)
        pool.close()
    return mapped


def pipe_parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """ alternative parallel map

    source: http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class
    """
    def spawn(f):
        def fun(q_in, q_out):
            while True:
                i, x = q_in.get()
                if i is None:
                    break
                q_out.put((i, f(x)))
        return fun

    if nprocs == 1:
        return map(f, X)
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(f), args=(q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]
