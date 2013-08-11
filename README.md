dooML
=====

Slow machine learning algorithms, because good data analysis takes time.

Under construction, check back in the future!

Transform Assumptions
=====================
1. A transform implements the fit and transform methods.
2. The fit method will be defined as: fit(self, X, y) or fit(self, X, y=None), with X a 2D numpy array, and y a 2D numpy array with the same number of rows as X or None.
3. The transform method will be defined as: def transform(self, X), with X a 2D numpy array.
4. The transform method will return a 2D numpy array, with the same number of rows as input data.
