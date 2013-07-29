"""Table of Contents
    -print_pickle
    -try_mkdir
    -quick_exists
    -quick_save
    -quick_load
    -file_cache
    -quick_cache
    -quick_write

"""
import gc
import cPickle as pickle
import shelve
from os import makedirs, path

import SETTINGS


def print_pickle(filename):
    """
    Prints the content of a pickle file.
    """
    with open(filename) as infile:
        x = pickle.load(infile)
        print(x)
    return x


def try_mkdir(directory):
    """ try to make directory
    """
    try:
        makedirs(directory)
    except OSError:
        pass


def quick_exists(directory, filename, extension):
    new_filename = path.join(directory, "{}.{}".format(filename, extension))
    return path.exists(new_filename)


def quick_save(directory, filename, obj):
    """Quickly pickle an object in a file.
    """
    try_mkdir(directory)
    gc.disable()
    new_filename = path.join(directory, "{}.pickle".format(filename))
    with open(new_filename, 'w') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)
    gc.enable()


def quick_load(directory, filename):
    """Quickly unpickle an object from a file.
    """
    new_filename = path.join(directory, "{}.pickle".format(filename))
    gc.disable()
    with open(new_filename) as infile:
        obj = pickle.load(infile)
    gc.enable()
    return obj


def file_cache(directory, unique_name, func, *args, **kwargs):
    """ a generic file-based cache that can be used to construct more complex caches. useful for storing bigger objects than a dictionary cache.
    """
    try:
        return quick_load(directory, unique_name)
    except:
        print("Cache Miss: {}".format(unique_name))
        result = func(*args, **kwargs)
        quick_save(directory, unique_name, result)
        return result


def dictionary_cache(filename, unique_name, func, *args, **kwargs):
    """ a generic file-based cache that can be used to construct more complex caches. useful for storing more objects than a file cache. NOT MULTIPROCESSING SAFE.
    """
    assert isinstance(unique_name, str), unique_name
    try:
        try:
            s = shelve.open(filename, flag='r', protocol=pickle.HIGHEST_PROTOCOL)
            return s[unique_name]
        finally:
            s.close()
    except:
        result = func(*args, **kwargs)
        s = shelve.open(filename, flag='c', protocol=pickle.HIGHEST_PROTOCOL)
        s[unique_name] = result
        s.close()
        return result


def quick_cache(unique_name, func, *args, **kwargs):
    """ an easy to use fast cache
    """
    return file_cache(SETTINGS.QUICK_CACHE.DIRECTORY, unique_name, func, *args, **kwargs)


def quick_write(directory, filename, obj):
    """ quickly write an object as a string to a file
    """
    try_mkdir(directory)
    new_filename = path.join(directory, "{}.txt".format(filename))
    with open(new_filename, 'w') as outfile:
        outfile.write(str(obj))


def machine_cache(filename, clf, *args, **kwargs):
    """ caches a machine fit to data
    """
    unique_name = repr(clf)
    return dictionary_cache(filename, unique_name, clf.fit, *args, **kwargs)
