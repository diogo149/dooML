"""Table of Contents
    -func_name
    -decorator_template
    -default_catcher
    -log
    -timer
    -trace_error
    -ignore_args
    -memmap

    -decorate_fit
    -decorate_transform
    -decorate_trn
    -memmap_trn
    -timer_trn
    -log_trn
    -trace_error_trn

    -humanize
"""

from __future__ import print_function
import numpy as np

from time import time
from pdb import set_trace
from tempfile import TemporaryFile

import SETTINGS

# these are imported so that they can be imported from this file
from helper.humanize import humanize


def func_name(func):
    try:
        return func.func_name
    except AttributeError:
        return "unnamed_function"


def decorator_template(func):

    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    wrapped.func_name = func_name(func)
    return wrapped


def default_catcher(default_value):
    """
    If the decorated function fails we instead use a decorated value.
    """
    def decorator(func):
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                return default_value
        wrapped.func_name = func_name(func)
        return wrapped
    return decorator


def log(func):
    """
    Logs input, output, and time takes of a decorated function.
    """
    def wrapped(*args, **kwargs):
        if SETTINGS.DECORATORS.LOG:
            print("Calling function: " + func.func_name)
            print("  Arguments:")
            for arg in args:
                print("    {}".format(arg))
            print("  Keyword Arguments:")
            for k, v in kwargs.items():
                print("    {}: {}".format(k, v))
            start_time = time()
        output = func(*args, **kwargs)
        if SETTINGS.DECORATORS.LOG:
            print("Returning function: " + func.func_name)
            print("Took {} seconds".format(time() - start_time))
            print("  Output:")
            print("    {}\n".format(output))

        return output

    wrapped.func_name = func_name(func)
    return wrapped


def timer(func):
    """
    Times the decorated function.
    """

    def wrapped(*args, **kwargs):
        start_time = time()
        output = func(*args, **kwargs)
        print("Function {} took {} seconds.".format(func_name(func), time() - start_time))
        return output

    wrapped.func_name = func_name(func)
    return wrapped


def trace_error(func):
    """
    If decorated function throws an exception, then the python debugger is started.
    """

    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("{} in {}: {}".format(e.__class__, func_name(func), e.message))
            set_trace()
            return func(*args, **kwargs)

    wrapped.func_name = func_name(func)
    return wrapped


def ignore_args(func):

    def wrapped(*args, **kwargs):
        return func()

    wrapped.func_name = func_name(func)
    return wrapped


def memmap(func):
    """ converts numpy array inputs into memmaps. useful for parallelization.
    """

    def memmap_and_tmp_file(arg):
        tmp_file = TemporaryFile()
        dtype = arg.dtype if SETTINGS.DECORATORS.MEMMAP_DTYPE is None else SETTINGS.DECORATORS.MEMMAP_DTYPE
        new_np = np.memmap(tmp_file, dtype=dtype, shape=arg.shape)
        new_np[:] = arg[:]
        return new_np, tmp_file

    def wrapped(*args, **kwargs):
        tmp_files = []
        try:
            new_args = []
            new_kwargs = {}
            for arg in args:
                if isinstance(arg, np.ndarray):
                    new_np, tmp_file = memmap_and_tmp_file(arg)
                    tmp_files.append(tmp_file)
                    new_args.append(new_np)
                else:
                    new_args.append(arg)
            for k, v in kwargs.items():
                if isinstance(v, np.npdarray):
                    new_np, tmp_file = memmap_and_tmp_file(v)
                    tmp_files.append(tmp_file)
                    new_kwargs[k] = new_np
                else:
                    new_kwargs[k] = v
            return func(*new_args, **new_kwargs)
        finally:
            for tmp_file in tmp_files:
                tmp_file.close()

    wrapped.func_name = func_name(func)
    return wrapped


def decorate_fit(decorator, trn):
    """ applies a decorator to the fit method of a transform
    """
    trn.fit = decorator(trn.fit)
    return trn


def decorate_transform(decorator, trn):
    """ applies a decorator to the transform method of a transform
    """
    trn.transform = decorator(trn.transform)
    return trn


def decorate_trn(decorator, trn):
    """ applies a decorator to both the fit and transform methods of a transform
    """
    return decorate_fit(decorator, decorate_transform(decorator, trn))


def memmap_trn(trn):
    """ Temporarily memmaps the X and y values. Useful for saving memory during parallel computation.
    """
    return decorate_trn(memmap, trn)


def timer_trn(trn):
    """ Times fit and transform methods
    """
    return decorate_trn(timer, trn)


def log_trn(trn):
    """ logs input and output of fit and transform methods
    """
    return decorate_trn(log, trn)


def trace_error_trn(trn):
    """ starts the python debugger if an exception is thrown in the fit or transform methods
    """
    return decorate_trn(trace_error, trn)


if __name__ == "__main__":
    @default_catcher(3)
    def doo():
        raise Exception

    print(doo())

    @log
    def sample(a, b, c, d, e, f):
        return [1, 2, 3, 4, 5]

    sample(1, 2, 3, 4, e=8, f=9)
    SETTINGS.DECORATORS.LOG = True
    sample(1, 2, 3, 4, e=8, f=9)

    @trace_error
    def goo():
        raise Exception("hello, world")

    goo()
