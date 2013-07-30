"""Table of Contents
    -decorator_template
    -default_catcher
    -log
    -timer
    -trace_error
    -ignore_args
"""

from __future__ import print_function
from time import time
import SETTINGS
from pdb import set_trace


def decorator_template(func):

    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    try:
        wrapped.func_name = func.func_name
    except AttributeError:
        pass
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
        try:
            wrapped.func_name = func.func_name
        except AttributeError:
            pass
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

    try:
        wrapped.func_name = func.func_name
    except AttributeError:
        pass
    return wrapped


def timer(func):
    """
    Times the decorated function.
    """

    def wrapped(*args, **kwargs):
        start_time = time()
        output = func(*args, **kwargs)
        print("Function {} took {} seconds.".format(func.func_name, time() - start_time))
        return output

    try:
        wrapped.func_name = func.func_name
    except AttributeError:
        pass
    return wrapped


def trace_error(func):
    """
    If decorated function throws an exception, then the python debugger is started.
    """

    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("{} in {}: {}".format(e.__class__, func.func_name, e.message))
            set_trace()
            return func(*args, **kwargs)

    try:
        wrapped.func_name = func.func_name
    except AttributeError:
        pass
    return wrapped


def ignore_args(func):

    def wrapped(*args, **kwargs):
        return func()

    try:
        wrapped.func_name = func.func_name
    except AttributeError:
        pass
    return wrapped

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
