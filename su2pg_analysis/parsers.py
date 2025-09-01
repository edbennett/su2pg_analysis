"""
Functions for parsing HiRep output files.
"""

import re

import numpy as np


class ParseError(ValueError):
    """
    Errors generated during parsing.
    """


def add_metadatum(metadata, key, value):
    """
    Add a given key-value pair to a metadata store,
    ensuring that it is consistent with what is already there.
    """
    if key in metadata:
        if metadata[key] != value:
            raise ParseError(f"Inconsistent data found for key {key}")
    else:
        metadata[key] = value


def get_time(line, skip=0):
    """
    Given a HiRep log line indicating a duration,
    return that duration as a floating-point number of seconds

    skip: number of extra fields printed at the end of the line to strip out.
    """
    line = " ".join(line.split()[: len(line.split()) - skip])
    if not line.endswith("usec]"):
        raise ParseError(f"No time found in line {line}")
    split_line = line.split()
    seconds = int(split_line[-4].lstrip("["))
    microseconds = int(split_line[-2])
    return seconds + microseconds / 1e6


def freeze(data):
    """
    Take a data structure of lists and dicts of lists,
    and turn every list into a numpy array.
    """
    frozen_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            frozen_data[key] = np.array(value)
        elif isinstance(value, dict):
            frozen_data[key] = freeze(value)
        else:
            frozen_data[key] = value
    return frozen_data


def complete_trajectory(data, datum):
    """
    Take the new datum for the current trajectory
    and append its components to the lists of data for all trajectories.
    """
    if datum is None:
        return
    if len(data["trajectory"]) > 0 and datum["trajectory"] <= data["trajectory"][-1]:
        raise ParseError("Run goes backwards")
    if "filename" not in datum and "save_time" not in datum:
        datum = {"filename": None, "save_time": None, **datum}
    if set(data.keys()) != set(datum.keys()):
        raise ParseError(f"Inconsistent data for trajectory {datum['trajectory']}")
    for key, value in datum.items():
        data[key].append(value)


def check_common(line, metadata):
    """
    Check for line formats that appear in all HiRep output files,
    and process them into the metadata dict.
    """
    if line.startswith("[SYSTEM][0]Gauge group"):
        group_family, num_colors = line.split()[-1].rstrip(")").split("(")
        add_metadatum(metadata, "group_family", group_family)
        add_metadatum(metadata, "num_colors", int(num_colors))
    if line.startswith("[GEOMETRY_INIT][0]Global size is"):
        dimensions = map(int, line.split()[-1].split("x"))
        for key, dimension in zip(["nt", "nx", "ny", "nz"], dimensions):
            add_metadatum(metadata, key, dimension)


def read_pg(filename):
    """
    Read a HiRep pure gauge generation log file.
    Return metadata about the ensemble,
    and plaquette values and timings as the ensemble is generated.
    """
    metadata = {}
    data = {
        "trajectory": [],
        "plaquette": [],
        "generation_time": [],
        "filename": [],
        "save_time": [],
    }
    current_datum = None

    with open(filename, "r", encoding="utf-8") as pg_file:
        for line in pg_file:
            check_common(line, metadata)
            if line.startswith("[INIT][0]beta="):
                add_metadatum(metadata, "beta", float(line.split("=")[-1]))
            if line.startswith("[INIT][0]nhb="):
                num_heatbath = int(line.split()[0].split("=")[-1])
                add_metadatum(metadata, "num_heatbath", num_heatbath)
                num_overrelaxed = int(line.split("=")[-1])
                add_metadatum(metadata, "num_overrelaxed", num_overrelaxed)
            if line.startswith("[MAIN][0]Thermalized"):
                add_metadatum(metadata, "num_thermalization", int(line.split()[1]))
                add_metadatum(metadata, "thermalization_time", get_time(line))
            if line.startswith("[MAIN][0]Trajectory"):
                if line.endswith("...\n"):
                    complete_trajectory(data, current_datum)
                    trajectory_index = int(line.split()[-1].strip("#."))
                    current_datum = {"trajectory": trajectory_index}
                if "generated" in line:
                    add_metadatum(current_datum, "generation_time", get_time(line))
            if line.startswith("[MAIN][0]Plaquette"):
                add_metadatum(current_datum, "plaquette", float(line.split()[-1]))
            if (
                line.startswith("[IO][0]Configuration")
                and line.split()[2] == "saved"
                and current_datum is not None
            ):
                add_metadatum(current_datum, "filename", line.split()[1].strip("[]"))
                add_metadatum(current_datum, "save_time", get_time(line))

        complete_trajectory(data, current_datum)

    metadata["num_trajectories"] = max(data["trajectory"]) - min(data["trajectory"]) + 1
    return metadata, freeze(data)


def complete_configuration(data, datum):
    """
    Take the new datum for the current configuration
    and append its components to the lists of data for all configurations.
    """
    for key, values in data.items():
        if isinstance(values, list):
            values.append(datum[key])
        elif isinstance(values, dict):
            if values == {}:
                for subkey, subvalue in datum[key].items():
                    values[subkey] = [subvalue]
            else:
                value = datum[key]
                assert values.keys() == value.keys()
                for subkey, subvalue in value.items():
                    values[subkey].append(subvalue)
        else:
            raise ParseError


def check_observable_common(line, metadata, current_datum):
    """
    Check for output lines that appear in all HiRep observable computation logs,
    and process their contents into the metadata or current_datum.
    """
    check_common(line, metadata)
    trajectory = None
    if line.startswith("[MAIN][0]Configuration from "):
        (trajectory,) = map(int, re.match(".*n([0-9]+)$", line.split()[-1]).groups())
        beta = float(re.match(".*b([0-9]+[.][0-9]+)", line.split()[-1]).groups()[0])
        add_metadatum(metadata, "beta", beta)
    if line.startswith("[IO][0]Configuration [") and line.split()[2] == "read":
        current_datum["read_time"] = get_time(line, skip=1)

    if line.startswith("[IO][0]Configuration ["):
        plaquette = float(line.split()[-1].split("=")[-1])
        current_datum["plaquette"] = {"complete": plaquette}
    if line.startswith("[PLAQ][0]Plaq("):
        mu_nu = tuple(map(int, line[14:17].split(",")))
        split_line = line.split()
        real_plaquette = float(split_line[3])
        imaginary_plaquette = float(split_line[5])
        current_datum["plaquette"][mu_nu] = real_plaquette + imaginary_plaquette * 1j
    return trajectory


def check_representation(line, metadata):
    if not line.startswith("[SYSTEM][0]MACROS=-D"):
        return
    representation = None
    if "REPR_FUNDAMENTAL" in line:
        representation = "fun"
    if "REPR_ADJOINT" in line:
        if representation:
            raise NotImplementedError("Multi-representation correlators not supported.")
        representation = "adj"
    if "REPR_SYMMETRIC" in line:
        if representation:
            raise NotImplementedError("Multi-representation correlators not supported.")
        representation = "sym"
    if "REPR_ANTISYMMETRIC" in line:
        if representation:
            raise NotImplementedError("Multi-representation correlators not supported.")
        representation = "asy"

    if "representation" in metadata and metadata["representation"] != representation:
        raise ValueError("Inconsistent representations.")

    metadata["representation"] = representation


def read_correlators(filename):
    """
    Read a HiRep meson correlation function log.
    Return metadata about the ensemble,
    correlation function values for each channel,
    and timings to compute these.
    """
    metadata = {}
    data = {
        "plaquette": {},
        "trajectory": [],
        "correlators": {},
        "read_time": [],
        "analysis_time": [],
    }
    current_datum = None
    trajectory = None
    with open(filename, "r", encoding="utf-8") as corr_file:
        for line in corr_file:
            trajectory = (
                check_observable_common(line, metadata, current_datum) or trajectory
            )
            check_representation(line, metadata)
            if line.startswith("[MAIN][0]Configuration from "):
                current_datum = {"trajectory": trajectory, "correlators": {}}

            if line.startswith("[MAIN][0]conf #"):
                split_line = line.split()
                valence_mass = float(split_line[2].split("=")[-1])
                source_type = split_line[3]
                connection = split_line[4]
                channel = split_line[5].rstrip("=")
                correlator = np.array(list(map(float, split_line[6:])))
                if channel.endswith("_im"):
                    current_datum["correlators"][
                        valence_mass, source_type, connection, channel[:-3]
                    ] = (
                        current_datum["correlators"][
                            valence_mass, source_type, connection, channel[:-3]
                        ]
                        + correlator * 1j
                    )
                else:
                    if channel.endswith("_re"):
                        channel = channel[:-3]
                    current_datum["correlators"][
                        valence_mass, source_type, connection, channel
                    ] = correlator
            if line.startswith("[MAIN][0]Configuration #") and line.split()[2:4] == [
                "analysed",
                "in",
            ]:
                current_datum["analysis_time"] = get_time(line)
                complete_configuration(data, current_datum)

    return metadata, freeze(data)


def read_flows(filename):
    """
    Read a HiRep gradient flow log.
    Return metadata about the ensemble,
    values for the energy density and topological charge for each flow time,
    and timings to compute these.
    """
    metadata = {}
    data = {
        "plaquette": {},
        "trajectory": [],
        "topological_charges": [],
        "plaquette_energy_densities": [],
        "clover_energy_densities": [],
        "flow_times": [],
        "read_time": [],
        "analysis_time": [],
    }
    current_datum = None
    trajectory = None
    with open(filename, "r", encoding="utf-8") as wflow_file:
        for line in wflow_file:
            trajectory = (
                check_observable_common(line, metadata, current_datum) or trajectory
            )
            if line.startswith("[MAIN][0]Configuration from "):
                current_datum = {
                    "trajectory": trajectory,
                    "topological_charges": [],
                    "plaquette_energy_densities": [],
                    "clover_energy_densities": [],
                    "flow_times": [],
                }

            if line.startswith("[WILSONFLOW][0]WF"):
                split_line = line.split()
                current_datum["flow_times"].append(float(split_line[3]))
                current_datum["plaquette_energy_densities"].append(float(split_line[4]))
                current_datum["clover_energy_densities"].append(float(split_line[6]))
                current_datum["topological_charges"].append(float(split_line[8]))
            if (
                line.startswith(
                    "[TIMING][0]Wilson Flow evolution and measurements for configuration"
                )
                and line.split()[8] == "done"
            ):
                current_datum["analysis_time"] = get_time(line)
                complete_configuration(data, current_datum)

    (unique_flow_times,) = list(
        set(tuple(flow_times) for flow_times in data["flow_times"])
    )
    data["flow_times"] = list(unique_flow_times)

    return metadata, freeze(data)
