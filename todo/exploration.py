"""exploration notes:

    train[col_name].dtype

    train[col_name].describe()

    for counts:
        sorted(Counter(train[col_name]).items())
        sorted(Counter(train[col_name]).items(), key=lambda x: x[1])
        train[col_name].value_counts()


    for unique:
        train["title"].unique().shape

    to plot numeric:
        pylab.plot(sorted(train[col_name]))
        pylab.plot(sorted(train[col_name].dropna()))
        pylab.plot(sorted(np.log(train[col_name] + 1)))
        pylab.plot(sorted(np.log(train[col_name].dropna() + 1)))
        pylab.plot(sorted(np.log(train[col_name] + 1e-6))
        pylab.plot(sorted(np.log(train[col_name].dropna() + 1e-6))
        pylab.show()

    for missing:
        ???
"""
