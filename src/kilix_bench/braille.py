"""Braille decomposition entry points; shared protocol lives in decomposition."""
from kilix_bench.decomposition import quality, check_reference
from kilix_bench import decomposition


def read_experiment(path):
    return decomposition.read_experiment(path, family="braille")


def inputs(args):
    return decomposition.inputs(args, family="braille")


def measure(args):
    return decomposition.measure(args, family="braille")
