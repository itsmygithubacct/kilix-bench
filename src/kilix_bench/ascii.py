"""ASCII spatial and repertoire experiments through the public native API."""
from kilix_bench import decomposition


def read_experiment(path):
    return decomposition.read_experiment(path, family="ascii")


def inputs(args):
    return decomposition.inputs(args, family="ascii")


def measure(args):
    return decomposition.measure(args, family="ascii")
