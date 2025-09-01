#!/usr/bin/env python3

"""
Read and write files
"""

import pyerrors as pe


def name_ensemble(metadata):
    """
    Generate a name for an ensemble with specified metadata,
    so that pyerrors correctly considers
    correlations between observables from the same ensemble.
    """
    return "{group_family}({num_colors})_pg_beta{beta}_{nt}x{nx}x{ny}x{nz}".format(
        **metadata
    )


def merge_metadata(old_metadata, new_metadata):
    """
    Update an existing set of metadata with some extra data,
    ensuring that any common keys have consistent values.
    """
    for key, value in new_metadata.items():
        if key in old_metadata:
            if old_metadata[key] != value:
                raise ValueError("Inconsistent metadata.")
        else:
            old_metadata[key] = value


def get_data(filenames):
    """Read data from filenames, and arrange into a hierarchical structure."""
    data = {}
    for filename in filenames:
        read_datum = pe.input.json.load_json(filename, verbose=False, full_output=True)
        metadata = read_datum["description"]
        if "description" in metadata:
            metadata = metadata["description"]
        ensemble_id = name_ensemble(metadata)
        if ensemble_id not in data:
            data[ensemble_id] = metadata
        else:
            merge_metadata(data[ensemble_id], metadata)
        obs_datum = read_datum["obsdata"][0]
        if obs_datum.tag in data[ensemble_id]:
            raise ValueError(f"Duplicate data for {obs_datum.tag} for {ensemble_id}")

        obs_datum.gamma_method()
        data[ensemble_id][obs_datum.tag] = obs_datum

    all_keys = set(key for datum in data.values() for key in datum.keys())
    reordered_data = {key: [] for key in all_keys}
    for datum in data.values():
        for key in all_keys:
            reordered_data[key].append(datum.get(key))

    return reordered_data
