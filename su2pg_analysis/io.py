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


def channelise_tag(bare_tag, channel):
    if not channel:
        return bare_tag

    return f"{channel}_{bare_tag}"


def label_obs_channels(datum):
    metadata = datum["description"]
    channel = metadata.pop("channel", None)
    datum["obsdata"] = {
        channelise_tag(bare_tag, channel): value
        for bare_tag, value in datum["obsdata"].items()
    }


def get_data(filenames):
    """Read data from filenames, and arrange into a hierarchical structure."""
    data = {}
    for filename in filenames:
        read_datum = pe.input.json.load_json_dict(
            filename, verbose=False, full_output=True
        )
        label_obs_channels(read_datum)
        metadata = read_datum["description"]
        ensemble_id = name_ensemble(metadata)
        if ensemble_id not in data:
            data[ensemble_id] = metadata
        else:
            merge_metadata(data[ensemble_id], metadata)
        for tag, value in read_datum["obsdata"].items():
            if tag in data[ensemble_id]:
                raise ValueError(f"Duplicate data for {tag} for {ensemble_id}")

            value.gamma_method()
            data[ensemble_id][tag] = value

    all_keys = set(key for datum in data.values() for key in datum.keys())
    reordered_data = {key: [] for key in all_keys}
    for datum in data.values():
        for key in all_keys:
            reordered_data[key].append(datum.get(key))

    return reordered_data
