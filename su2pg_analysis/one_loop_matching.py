#!/usr/bin/env python3

from argparse import ArgumentParser

from numpy import pi
import pyerrors as pe

from .io import get_data


def get_args():
    parser = ArgumentParser(
        description="Normalise decay constants with one-loop matching coefficients",
    )
    parser.add_argument(
        "--spectral_observable_data",
        required=True,
        help="File containing computed spectral quantities for a given symmetry channel",
    )
    parser.add_argument(
        "--plaquette_data",
        required=True,
        help="File containing computed mean plaquette.",
    )
    parser.add_argument(
        "--output_filename",
        default=None,
        help="Where to output results",
    )
    return parser.parse_args()


def get_channel(data):
    if "ps_amplitude" in data:
        return "ps"
    elif "v_amplitude" in data:
        return "v"
    else:
        raise NotImplementedError("Only PS and V channels supported currently.")


def quad_casimir(group_family, num_colors, representation):
    """
    Return the eigenvalue of
    the quadratic Casimir operator of the given representation of the given group.
    """
    if group_family != "SU":
        raise NotImplementedError("This code only supports SU(N) currently.")

    if representation == "fun":
        # Per https://doi.org/10.21468/SciPostPhysLectNotes.21 Eq. (5)
        return (num_colors**2 - 1) / (2 * num_colors)
    if representation == "adj":
        # Per https://doi.org/10.21468/SciPostPhysLectNotes.21 Eq. (45)
        return num_colors
    return NotImplementedError(
        "Currently only fundemantal and adjoint representations supported."
    )


def renormalise_decay_constant(plaquette_data, spectrum_data):
    plaquette = plaquette_data["plaquette"][0]
    amplitude = spectrum_data["obsdata"]["amplitude"]
    beta = plaquette_data["beta"][0]
    assert beta == spectrum_data["description"]["beta"]

    # Z_V = 1 + C_F \Delta \tilde{g}^2 / {16 \pi^2}
    # \tilde{g} \langle P \rangle = 8 / \beta
    # \Delta = \Delta_{\Sigma_1} + \Delta_{\gamma_mu}
    # Per 1712.04220, Eq. (6.8)
    delta_sigma_one = -12.82
    delta_gmu = -7.75
    casimir = quad_casimir(
        spectrum_data["description"]["group_family"],
        spectrum_data["description"]["num_colors"],
        spectrum_data["description"]["representation"],
    )
    Z_coefficient = 1 + casimir * (delta_sigma_one + delta_gmu) * (
        8 / beta / plaquette
    ) / (16 * pi**2)
    result = amplitude * Z_coefficient
    result.gamma_method()
    return result


def main():
    args = get_args()
    plaquette_data = get_data([args.plaquette_data])
    spectrum_data = pe.input.json.load_json_dict(
        args.spectral_observable_data, verbose=False, full_output=True
    )

    channel = spectrum_data["description"]["channel"]
    result = renormalise_decay_constant(plaquette_data, spectrum_data)

    if args.output_filename is None:
        print(f"{channel}_decay_const: {result}")
    else:
        pe.input.json.dump_dict_to_json(
            {
                f"{channel}_decay_const": result,
            },
            args.output_filename,
            description=spectrum_data["description"],
        )


if __name__ == "__main__":
    main()
