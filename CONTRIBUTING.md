# Contributing

We welcome new contributions towards ANNchor. Feel free to get in touch if you are thinking about contributing, and follow the rough guidelines below.

## Issues

If you find any issues/bugs in the software, please [file an issue](https://github.com/gchq/annchor/issues/new).
If possible, provide details on how to reproduce the issue, and we'll try our best to get it fixed.

## Code

You are welcome to contribute towards the ANNchor codebase (e.g bug-fixes, docs, examples, new features).
Please fork the project, make your changes, and submit a pull request to the main repository.
Make sure to include clear details about the purpose of your pull request.

### Code formatting

ANNchor uses the [black code formatter](https://github.com/python/black), which can be installed
via pip (`pip install black`). Simply install and run in the project root. Note that we run black with a maximum line length of 79.

```bash
black --line-length 79 ./
```

### Testing

Make sure that any contributions pass the unit tests in `annchor/tests`. First, install the testing dependencies:

```shell
$ pip install .
$ pip install pytest networkx
```

Then, run `pytest`:

```shell
$ pytest .
```

To collect and view code coverage:

```shell
$ pip install coverage
$ coverage run
$ coverage report
```


## Documentation

To build the documentation, first install the pinned dependencies:

```shell
$ pip install -r doc/requirements.txt --no-deps
```

Then run the following:

```shell
$ python3 -m sphinx -b html doc/source doc/build
```

The built documentation can then be opened at [`doc/build/index.html`](doc/build/index.html).
