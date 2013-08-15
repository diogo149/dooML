"""Table of Contents
    -parmap
    -pipe_parmap
    -parfor
    -random_parmap_helper
    -random_parmap
    -joblib_parmap
    -joblib_run
    -no_pickle_parmap
    -pmap
"""

import multiprocessing

from pickle import PicklingError
from functools import partial
from joblib import Parallel, delayed

import SETTINGS
from decorators import deprecated
from utils import random_seed


@deprecated
def parmap(func, in_vals, args=(), kwargs={}, n_jobs=1):
    """ easy parallel map, but it pickles input arguments and thus can't be used for dynamically generated functions.
    """
    assert isinstance(n_jobs, int)
    assert n_jobs >= -1
    if args or kwargs:
        new_func = partial(func, *args, **kwargs)
    else:
        new_func = func
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


@deprecated
def pipe_parmap(func, X, n_jobs=1):
    """ alternative parallel map

    source: http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class
    """
    def spawn(func):
        def fun(q_in, q_out):
            while True:
                i, x = q_in.get()
                if i is None:
                    break
                q_out.put((i, func(x)))
        return fun

    if n_jobs == 1:
        return map(func, X)
    elif n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(func), args=(q_in, q_out)) for _ in range(n_jobs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(n_jobs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def parfor(func, num_times, args=(), kwargs={}, n_jobs=1):
    """ run a function multiple times with the same input in parallel
    """
    def wrapped(i):
        random_seed(i)
        return func(*args, **kwargs)
    return pipe_parmap(wrapped, range(num_times), n_jobs=n_jobs)


def random_parmap_helper(func, args, kwargs, item):
    """ helper method for random_parmap
    """
    idx, in_val = item
    random_seed(idx)
    return func(in_val, *args, **kwargs)


def random_parmap(func, in_vals, args=(), kwargs={}, n_jobs=1):
    """ parallel map with different random seeds for each value
    """
    return parmap(random_parmap_helper, zip(range(len(in_vals)), in_vals), [func, args, kwargs], n_jobs=n_jobs)


def joblib_parmap(func, generator):
    """ parallel map using joblib, but it pickles input arguments and thus can't be used for dynamically generated functions.
    """
    new_func = delayed(func)
    return joblib_run(new_func(item) for item in generator)


def joblib_run(delayed_generator):
    """ runs a generator of joblib tasks
    NOTE: the functions run do not have to be homogeneous, you can make arbitrary generators with whatever functions as long as they are pickle-able
    """
    return Parallel(n_jobs=SETTINGS.PARALLEL.JOBS, verbose=SETTINGS.PARALLEL.JOBLIB_VERBOSE, pre_dispatch=SETTINGS.PARALLEL.JOBLIB_PRE_DISPATCH)(delayed_generator)


def no_pickle_parmap(func, X):
    """ alternative parallel map that allows for unpicklable items by using pipes

    source: http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class
    """
    def spawn(func):
        def fun(q_in, q_out):
            while True:
                i, x = q_in.get()
                if i is None:
                    break
                q_out.put((i, func(x)))
        return fun

    n_jobs = multiprocessing.cpu_count()

    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(func), args=(q_in, q_out)) for _ in range(n_jobs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(n_jobs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def pmap(func, generator, *args, **kwargs):
    """ parallel map that only parallelizes if not already within a pmap
    """
    new_func = partial(func, *args, **kwargs) if args or kwargs else func
    if SETTINGS.PARALLEL.PMAP:
        return map(new_func, generator)
    else:
        try:
            SETTINGS.PARALLEL.PMAP = True
            return joblib_parmap(new_func, generator)
        except PicklingError:
            return no_pickle_parmap(new_func, generator)
        finally:
            SETTINGS.PARALLEL.PMAP = False
