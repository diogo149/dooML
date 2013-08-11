"""Table of Contents:
    -FeatureStore
    -FeatureNode
    -FeatureDependency
    -FeatureDB

implementation notes:
    -store nodes with same name in same file with backup open or equivalent with joblib
    -need ability to draw / list tree (?)
    -for node file, keep:
        -fit_time
        -transformer
        -loss (if "prediction" in tags)
    -CV transform
"""

import sqlite3

import SETTINGS
from utils import all_iterable, random_seed, bool_to_int
from storage import joblib_load, joblib_save


class FeatureStore(object):

    """ class for operations on collections of FeatureNodes
    """

    def __init__(self):
        """ initialize feature store
        """
        self.db = FeatureDB()

    def query_basename_id(self, basename):
        """ returns a collection of ids given a basename
        """
        return self.db.query_basename_id(basename)

    def node_id(self, label):
        """ returns an id given a label
        """
        return self.db.node_id(label)

    def query_tag(self, tag):
        """ returns a collection of ids given a tag
        """
        return self.db.query_tag(tag)

    def tag_nodes(self, new_tag, ids):
        """ add a new tag to an id. the tag cannot be one that currently exists to keep consistency.
        """
        # TODO

    def add(self, node):
        """ adds a FeatureNode to the store, but doesn't yet transform the data
        """
        return self.db.add_node(node)

    def node(self, node_id):
        """ returns a FeatureNode corresponding to the input id
        """
        return self.db.node(node_id)

    def prune_independent_of_id(self, id):
        """ removes the data stored on disk of all nodes not an ancestor of node with input id
        """
        # TODO

    def recursive_copy(self, feature_node, expand=True):
        """ TODO ??? how do i do this
        """

    def find_children(self, id):
        """ returns the children of the input id
        """
        # TODO

    def find_parents(self, id):
        """ returns the parents of the input id
        """
        # TODO make find_parents (to find ids) db function and batch_label function (to convert ids to labels efficiently)

    def input_node(self, data, name, tags=('raw', 'input'), data_name=None):
        if data_name is None:
            data_name = SETTINGS.FEATURE_STORE.DATA_NAME
        node = self.add(FeatureNode(name, None, tags=tags, X=FeatureDependency(), y=FeatureDependency(), cache=True, cache_output=True))
        node.save_field(data_name, "output", data)
        return node


class FeatureNode(object):

    def __init__(self, label, trn, tags, X, y, cache=True, cache_output=True):
        """
        warning: for user created nodes, have label be a string

        assumptions:
            -numbers fit in 32 bit float
            -if not cache and cache_output, output still isn't cached
        """
        self.basename, self.index = FeatureNode.parse_label(label)
        self.trn = trn
        self.tags = set(tags)
        self.X = X
        self.y = y
        self.cache = cache
        self.cache_output = cache_output

    def fit_transform(self, data_name=None):
        """ get the output of this feature for the training data
        """
        assert self.index is not None  # ???
        # if output doesnt exist
        # finds parents and calls to get their output
        # FIXME
        # auto memmap
        # auto CV
        # auto parallelize
        random_seed(self.index)

    def transform(self, data_name=None, fit_data_name=None):
        """ get the output of this feature for the data corresponding to data_name
        """
        # if transform doesn't exist
        # find parents and calls to get their output
        # FIXME
        # try to parallelize this?

    def save_field(self, data_name, field, value):
        """ saves a field corresponding to this node
        """
        if data_name is None:
            data_name = SETTINGS.FEATURE_STORE.DATA_NAME
        try:
            basename_data = joblib_load(data_name, self.basename)
        except IOError:
            basename_data = {}
        if self.index not in basename_data:
            basename_data[self.index] = {}
        basename_data[self.index][field] = value
        joblib_save(data_name, self.basename, basename_data)

    def load_field(self, data_name, field):
        """ loads a field corresponding to this node, or returns None if no such field exists
        """
        assert self.index is not None
        if data_name is None:
            data_name = SETTINGS.FEATURE_STORE.DATA_NAME
        try:
            data = joblib_load(data_name, self.basename)[self.index]
            return data[field]
        except:
            return None

    def save_transform(self, data_name=None):
        """ saves the transform to a file
        """
        return self.save_field(data_name, "raw_trn", self.trn)

    def load_transform(self, data_name=None):
        """ loads the transform from a file
        """
        self.trn = self.load_field(data_name, "raw_trn")
        return self.trn

    def prune_id(self):
        """ removes the data stored on disk for this node
        """
        # TODO

    @staticmethod
    def parse_label(label):
        if isinstance(label, str):
            basename, index = label, None
        elif isinstance(label, tuple):
            basename, index = label
        else:
            raise Exception
        assert FeatureNode.valid_name(basename)
        assert FeatureNode.valid_index(index)
        return basename, index

    @staticmethod
    def valid_name(name):
        """ True if name is valid
        """
        return isinstance(name, str) and len(name) <= 255 and name.isalnum()

    @staticmethod
    def valid_index(index):
        """ True if index is valid
        """
        return index is None or isinstance(index, int)

    @staticmethod
    def valid_label(label):
        """ True if label is valid
        """
        basename, index = label
        return FeatureNode.valid_name(basename) and FeatureNode.valid_index(index)

    @staticmethod
    def verify_node(node):
        """ AssertionError if node has invalid field
        """
        assert isinstance(node, FeatureNode)
        assert FeatureNode.valid_name(node.basename)
        assert FeatureNode.valid_index(node.index)
        assert all_iterable(node.tags, FeatureNode.valid_name)
        FeatureDependency.verify_dependency(node.X)
        FeatureNode.verify_dependency(node.y)


class FeatureDependency(object):

    def __init__(self, parent_labels=(), parent_tags=(), exclude_labels=(), exclude_tags=()):
        """in terms of priority: parent_labels > exclude_labels, exclude_tags > tags
        """
        self.parent_labels = FeatureDependency.parse_labels(parent_labels)
        self.exclude_labels = FeatureDependency.parse_labels(exclude_labels) - self.parent_labels
        self.exclude_tags = frozenset(exclude_tags)
        self.parent_tags = frozenset(parent_tags) - self.exclude_tags

    def __or__(self, other):
        """take the union of two dependencies
        """
        return FeatureDependency(self.parent_labels | other.parent_labels, self.parent_tags | other.parent_tags, self.exclude_labels | other.exclude_labels, self.exclude_tags | other.exclude_tags)

    def __and__(self, other):
        """ take the intersection of two dependencies
        """
        return FeatureDependency(self.parent_labels & other.parent_labels, self.parent_tags & other.parent_tags, self.exclude_labels & other.exclude_labels, self.exclude_tags & other.exclude_tags)

    def expand(self, node_id):
        """ creates a FeatureDependency where all the tags and excluded dependencies are converted into exact parent names
        """
        tmp = set(self.parent_labels)
        # TODO
        return tmp

    @staticmethod
    def verify_dependency(dep):
        """ AssertionError if dependency has invalid field
        """
        if dep is not None:
            assert all_iterable(dep.parent_labels, FeatureNode.valid_label)
            assert all_iterable(dep.exclude_tags, FeatureNode.valid_name)
            assert all_iterable(dep.parent_tags, FeatureNode.valid_name)
            assert all_iterable(dep.exclude_labels, FeatureNode.valid_label)

    @staticmethod
    def parse_labels(labels):
        """ parse multiple labels at the same time
        """
        return frozenset([(basename, index) if index is not None else (basename, -1) for basename, index in map(FeatureNode.parse_label, labels)])


class FeatureDB(object):

    TABLES = (
        ('NODES', 'NAME_ID INTEGER, CACHE INTEGER, CACHE_OUTPUT INTEGER'),
        ('NODE_TAGS', 'ID INTEGER, TAG_ID INTEGER'),
        ('PARENTS', 'ID INTEGER, PARENT_ID INTEGER'),
        ('EXCLUDE', 'ID INTEGER, EXCLUDE_ID INTEGER'),
        ('PARENT_TAGS', 'ID INTEGER, TAG_ID INTEGER'),
        ('EXCLUDE_TAGS', 'ID INTEGER, TAG_ID INTEGER'),
        ('PARENTS2', 'ID INTEGER, PARENT_ID INTEGER'),
        ('EXCLUDE2', 'ID INTEGER, EXCLUDE_ID INTEGER'),
        ('PARENT_TAGS2', 'ID INTEGER, TAG_ID INTEGER'),
        ('EXCLUDE_TAGS2', 'ID INTEGER, TAG_ID INTEGER'),
        ('TAGS', 'TAG VARCHAR(255) UNIQUE'),
        ('BASENAMES', 'BASENAME VARCHAR(255) UNIQUE'),
    )

    def __init__(self):
        """ setup database if it doesn't exist, and return a database connection
        """
        db_file = SETTINGS.FEATURE_STORE.DB
        self.conn = sqlite3.connect(db_file)
        for table_name, schema in FeatureDB.TABLES:
            self.create(table_name, schema)

    def sql(self, sql):
        """ run sql command
        """
        if SETTINGS.FEATURE_STORE.DEBUG_SQL:
            print sql
        try:
            c = self.conn.cursor()
            return c.execute(sql).fetchall()
        finally:
            c.close()
            self.conn.commit()

    def schema(self, table_name):
        return self.select('SQLITE_MASTER', 'SQL', ('TBL_NAME', table_name), ('TYPE', 'table'))[0][0]

    def column_names(self, table_name):
        schema = self.schema(table_name)
        col_schema = schema[schema.index('(') + 1:schema.index(')')]
        return [col.split()[0] for col in col_schema.split(',')]

    def create(self, table_name, schema):
        """ creates table with specified schema if it doesn't exist yet
        """
        self.sql('CREATE TABLE IF NOT EXISTS {} ({})'.format(table_name, schema))

    def insert(self, table_name, values, or_ignore=False):
        """ insert values into table
        """
        cmd = 'INSERT OR IGNORE' if or_ignore else 'INSERT'
        self.sql('{} INTO {} VALUES({})'.format(cmd, table_name, ",".join(map(repr, values))))

    def last_insert_rowid(self):
        """ get the rowid of the last insert
        """
        return self.sql('SELECT LAST_INSERT_ROWID()')[0][0]

    def select(self, table_name, values, *conditions):
        """ select values from table, before optional rowid
        """
        sqls = ['SELECT {} FROM {}'.format(values, table_name)]
        if conditions:
            sqls.append('WHERE')
            sqls.append(' AND '.join(['{}={}'.format(column, repr(value)) for column, value in conditions]))
        return self.sql(' '.join(sqls))

    def select_before(self, table_name, values, rowid, *conditions):
        """ select values from table, before rowid
        """
        sqls = ['SELECT {} FROM {}'.format(values, table_name)]
        sqls.append('WHERE ROWID<{}'.format(rowid))
        sqls.append(' AND '.join(['{}={}'.format(column, repr(value)) for column, value in conditions]))
        return self.sql(' '.join(sqls))

    def update(self, table_name, update_expr, *conditions):
        """ update values in table
        """
        sqls = ['UPDATE {} SET {}'.format(table_name, update_expr)]
        if conditions:
            sqls.append('WHERE')
            sqls += ['{}={}'.format(column, repr(value)) for column, value in conditions]
        return self.sql(' '.join(sqls))

    def get_id(self, table_name, value):
        """ gets the id of a unique value in a table with only one column, inserting it if it doesn't exist yet
        """
        column = self.column_names(table_name)[0]
        self.insert(table_name, (value,), or_ignore=True)
        return self.select(table_name, 'ROWID', (column, value))[0][0]

    def add_node(self, node):
        """ add each part of the node to the database and update the node's index
        """
        basename_id = self.basename_id(node.basename)
        self.insert('NODES', (basename_id, bool_to_int(node.cache), bool_to_int(node.cache_output)))
        node_id = self.last_insert_rowid()
        for tag_id in map(self.tag_id, node.tags):
            self.insert('NODE_TAGS', (node_id, tag_id))
        self.add_dependency(node_id, '', node.X)
        self.add_dependency(node_id, '2', node.y)
        node.index = self.node_index(node_id, basename_id)
        node.save_transform()
        return node

    def node(self, node_id):
        """ returns a FeatureNode given a node id
        """
        basename_id, cache, cache_output = self.select('NODES', '*', ('ROWID', node_id))
        index = self.node_index(node_id, basename_id)
        basename = self.basename(basename_id)
        label = (basename, index)
        tag_ids = self.tag_ids(node_id)
        tags = [self.tag(tag_id) for tag_id in tag_ids]
        X = self.dependency(node_id, '')
        y = self.dependency(node_id, '2')
        node = FeatureNode(label, None, tags, X, y, cache, cache_output)
        node.load_transform()
        return node

    def add_dependency(self, node_id, table_suffix, dep):
        """ add each part of the dependency to the specified table
        """
        for tag in dep.parent_tags:
            self.insert("PARENT_TAGS{}".format(table_suffix), (node_id, self.tag_id(tag)))
        for tag in dep.exclude_tags:
            self.insert("EXCLUDE_TAGS{}".format(table_suffix), (node_id, self.tag_id(tag)))
        for label in dep.parent_labels:
            self.insert("PARENTS{}".format(table_suffix), (node_id, self.node_id(label)))
        for label in dep.exclude_labels:
            self.insert("EXCLUDE{}".format(table_suffix), (node_id, self.node_id(label)))

    def dependency(self, node_id, table_suffix):
        """ load each part of the dependency from the specified table
        """
        parent_labels = [self.label(row[0]) for row in self.select("PARENTS{}".format(table_suffix), 'PARENT_ID', ('ID', node_id))]
        exclude_labels = [self.label(row[0]) for row in self.select("EXCLUDES{}".format(table_suffix), 'EXCLUDE_ID', ('ID', node_id))]
        parent_tags = [self.tag(row[0]) for row in self.select("PARENT_TAGS{}".format(table_suffix), 'TAG_ID', ('ID', node_id))]
        exclude_tags = [self.tag(row[0]) for row in self.select("EXCLUDE_TAGS{}".format(table_suffix), 'TAG_ID', ('ID', node_id))]
        return FeatureDependency(parent_labels, parent_tags, exclude_labels, exclude_tags)

    def label(self, node_id):
        """ returns a label given an id
        """
        basename_id = self.select('NODES', 'NAME_ID', ('ROWID', node_id))[0][0]
        index = self.node_index(node_id, basename_id)
        return self.basename(basename_id), index

    def tag_ids(self, node_id):
        """ returns tags of a node with input node id
        """
        return [row[0] for row in self.select("NODE_TAGS", "TAG_ID", ("ID", node_id))]

    def basename(self, basename_id):
        """ returns a basename given a basename id
        """
        return self.select('BASENAMES', 'BASENAME', ('ROWID', basename_id))[0][0]

    def tag(self, tag_id):
        """ returns a tag given a tag id
        """
        return self.select('TAGS', 'TAG', ('ROWID', tag_id))[0][0]

    def node_index(self, node_id, basename_id):
        """ return a node's index given it's id and basename
        """
        return sorted(self.query_basename_id(basename_id)).index(node_id)

    def basename_id(self, basename):
        """ gets the id of a basename, inserting it if it doesn't exist yet
        """
        return self.get_id('BASENAMES', basename)

    def tag_id(self, tag):
        """ gets the id of a tag, inserting it if it doesn't exist yet
        """
        return self.get_id('TAGS', tag)

    def node_id(self, label):
        """ returns an id given a label

        WARNING: indexing with negative numbers may cause weird behavior with parallel behavior; also note that then the node is loaded, it's parent will have the true index and not the relative (negative) one
        """
        basename, index = label
        basename_id = self.basename_id(basename)
        return sorted(self.query_basename_id(basename_id))[index]

    def query_basename_id(self, basename_id):
        """ returns a collection of ids given a basename_id
        """
        results = self.select('NODES', 'ROWID', ('NAME_ID', basename_id))
        return [result[0] for result in results]

    def query_tag(self, tag):
        """ returns a collection of ids that have the input tag
        """
        tag_id = self.tag_id(tag)
        return self.query_tag_id(tag_id)

    def query_tag_id(self, tag_id):
        """ returns a collection of ids given a tag_id
        """
        results = self.select('TAGS', 'ROWID', ('TAG', tag_id))
        return [result[0] for result in results]

    def close(self):
        """ close connection to the database
        """
        self.conn.commit()
        self.conn.close()
