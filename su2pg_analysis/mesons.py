#!/usr/bin/env python3

"""
Utilities for working with mesonic correlation functions.
"""

import pyerrors as pe

from .io import name_ensemble
from .parsers import read_correlators


IMPLEMENTED_CHANNELS = {"ps": ["g5"], "v": ["g1", "g2", "g3"]}
LATEX_DESCRIPTIONS = {"ps": r"\pi", "v": r"\rho"}


def get_channel(correlators, channel):
    """
    Pick out the single correlator with a specific symmetry `channel`
    from an ensemble of `correlators`.
    If multiple matches, or no matches, are found,
    raise an error.
    """
    (matching_correlator,) = [
        (key, correlator)
        for key, correlator in correlators.items()
        if key[-1] == channel
    ]
    return matching_correlator


def get_pyerrors_correlator(correlator, name, indices):
    """
    Take a 2D numpy array of correlator values,
    and turn it into a 1D pyerrors Corr object.
    """
    observables = [
        pe.Obs(
            [timeslice.real], [name], idl=[indices]
        )  # Force taking real part, so symmetrisation works correctly
        for timeslice in correlator.swapaxes(0, 1)
    ]
    return pe.Corr(observables)


def get_correlators_from_file(filename, input_channel):
    """
    Read the correlators in the specified filename,
    and filter for channels matching the channel descriptor specified.
    """
    metadata, data = read_correlators(filename)
    implemented_channels = IMPLEMENTED_CHANNELS[input_channel]
    component_correlators = [
        get_channel(data["correlators"], channel) for channel in implemented_channels
    ]
    assert len(set(tuple(key[:3]) for key, _ in component_correlators)) == 1
    metadata["mass"] = component_correlators[0][0][0]

    correlator = sum(
        get_pyerrors_correlator(
            component_correlator,
            name_ensemble(metadata),
            data["trajectory"],
        ).symmetric()
        for _, component_correlator in component_correlators
    ) / len(implemented_channels)
    correlator.gamma_method()
    metadata["channel"] = input_channel
    return metadata, correlator
