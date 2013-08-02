"""Table of Contents
    -print_pickle
    -try_mkdir
    -filename
    -quick_exists
    -quick_save
    -quick_load
    -file_cache
    -dictionary_cache
    -quick_cache
    -quick_write
    -machine_cache
    -related_filenames
    -modified_time
    -backup_open
    -quick_save2
    -quick_load2
    -quick_write2
    -quick_read2

"""
import gc
import cPickle as pickle
import shelve
from os import makedirs, path

import SETTINGS
from decorators import default_catcher


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


def filename(*args):
    """ returns a filename from components
    """
    extension = ""
    directory = "."
    if len(args) == 1:
        basename, = args
    elif len(args) == 2:
        basename, extension = args
    elif len(args) == 3:
        directory, basename, extension = args
    else:
        raise Exception
    if extension:
        extension = "." + extension
    return path.join(directory, filename + extension)


def quick_exists(directory, basename, extension):
    return path.exists(filename(directory, basename, extension))


def quick_save(directory, basename, obj):
    """Quickly pickle an object in a file.
    """
    try_mkdir(directory)
    gc.disable()
    new_filename = filename(directory, basename, "pickle")
    with open(new_filename, 'w') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)
    gc.enable()


def quick_load(directory, basename):
    """Quickly unpickle an object from a file.
    """
    new_filename = filename(directory, basename, "pickle")
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


def dictionary_cache(basename, unique_name, func, *args, **kwargs):
    """ a generic file-based cache that can be used to construct more complex caches. useful for storing more objects than a file cache. NOT MULTIPROCESSING SAFE.
    """
    assert isinstance(unique_name, str), unique_name
    basename = "{}.pickle".format(basename)
    try:
        try:
            s = shelve.open(basename, flag='r', protocol=pickle.HIGHEST_PROTOCOL)
            return s[unique_name]
        finally:
            s.close()
    except:
        result = func(*args, **kwargs)
        s = shelve.open(basename, flag='c', protocol=pickle.HIGHEST_PROTOCOL)
        s[unique_name] = result
        s.close()
        return result


def quick_cache(unique_name, func, *args, **kwargs):
    """ an easy to use fast cache
    """
    return file_cache(SETTINGS.QUICK_CACHE.DIRECTORY, unique_name, func, *args, **kwargs)


def quick_write(directory, basename, obj):
    """ quickly write an object as a string to a file
    """
    try_mkdir(directory)
    new_filename = filename(directory, basename, "txt")
    with open(new_filename, 'w') as outfile:
        outfile.write(str(obj))


def quick_read(directory, basename):
    """ quickly read a file
    """
    new_filename = filename(directory, basename, "txt")
    with open(new_filename) as infile:
        return infile.read()


def machine_cache(basename, clf, *args, **kwargs):
    """ caches a machine fit to data
    """
    unique_name = repr(clf)
    return dictionary_cache(basename, unique_name, clf.fit, *args, **kwargs)


def related_filenames(full_filename, num=2):
    """ returns num filenames similar to the input
    """
    basename, extension = path.splitext(full_filename)
    return ["{}_{}{}".format(basename, i, extension) for i in range(num)]


@default_catcher(None)
def modified_time(full_filename):
    """ returns the time a file was modified, or None if it doesn't exist
    """
    return path.getmtime(full_filename)


def backup_open(full_filename, flag='r', num=2):
    """ writes and reads to/from a group of files, so that data isn't lost in case of a failure
    """
    assert flag in 'war'
    filenames = related_filenames(full_filename, num)
    times = map(modified_time, filenames)
    indices = sorted(range(num), key=lambda x: times[x])
    idx = indices[-1] if flag == 'r' else indices[0]
    new_filename = filenames[idx]
    return open(new_filename, flag)


def quick_save2(directory, basename, obj):
    """Quickly pickle an object in a file, using backup_open
    """
    try_mkdir(directory)
    gc.disable()
    new_filename = filename(directory, basename, "pickle")
    with backup_open(new_filename, 'w') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)
    gc.enable()


def quick_load2(directory, basename):
    """Quickly unpickle an object from a file, using backup_open
    """
    new_filename = filename(directory, basename, "pickle")
    gc.disable()
    with backup_open(new_filename) as infile:
        obj = pickle.load(infile)
    gc.enable()
    return obj


def quick_write2(directory, basename, obj):
    """ quickly write an object as a string to a file
    """
    try_mkdir(directory)
    new_filename = filename(directory, basename, "txt")
    with backup_open(new_filename, 'w') as outfile:
        outfile.write(str(obj))


def quick_read2(directory, basename):
    """ quickly read a file
    """
    new_filename = filename(directory, basename, "txt")
    with backup_open(new_filename) as infile:
        return infile.read()
