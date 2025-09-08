#!/usr/bin/env python3

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pyerrors as pe

from .io import name_ensemble
from .mesons import get_pyerrors_correlator
from .parsers import read_flows


class UnreachableThreshold(ValueError):
    pass


def get_args():
    """
    Parse command-line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("wf_log", help="Wilson flow log file")
    parser.add_argument(
        "--output_filename",
        default=None,
        help="Where to output results",
    )
    parser.add_argument(
        "--W0",
        default=None,
        type=float,
        help="Reference scale: value of t d(t^2 E)/dt at which to solve for t^2 = w0",
    )
    parser.add_argument(
        "--plot_filename",
        default=None,
        help=(
            "Where to output a plot of the flow.If not specified, no plot is generated"
        ),
    )
    parser.add_argument("--plot_styles", default=None, help="Plot style file to use")
    return parser.parse_args()


def get_time_step(times):
    """
    Ensure that the flow times have a uniform separation,
    so that numerical derivatives work correctly
    without needing spline interpolation functions.
    """
    time_differences = times[1:] - times[:-1]
    if (
        max(abs(time_differences.mean() - time_differences))
        > 1e-4 * time_differences.mean()
    ):
        raise NotImplementedError(
            "w0 computation is only implemented for the non-adaptive Wilson flow."
        )
    return time_differences[0]


def quadratic_fit_form(params, t):
    a = params[0]
    b = params[1]
    c = params[2]
    return a * t**2 + b * t + c


def get_quadratic_left_solution(a, b, c):
    """
    Return the lower of the two solutiosn to the quadratic equation
    a x ^ 2 + b x + c = 0
    """
    # Could do this in a more sophisticated way by checking signs,
    # but this is probably quicker.
    return min(
        (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a),
        (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a),
    )


def compute_w0(times, energy_densities, name, indices, threshold):
    """
    Compute the value at which the quantity t d(t^2 E)/dt
    passes through the given threshold.
    Return this, along with the flow of the quantity itself.

    See 1203.4469 equations (2) and (3).
    """
    raw_t2E = times**2 * energy_densities  # t^2 E, numpy array
    t2E = get_pyerrors_correlator(raw_t2E, name, indices)  # t^2 E, pyerrors Corr
    dt2E_dt = t2E.deriv()  # d(t^2 E) / dt
    dt2E_dt.gamma_method()

    indices_to_fit = []
    t_dt2E_dt = []
    for index, (time, dt2E_dt_element) in enumerate(zip(times, dt2E_dt)):
        if dt2E_dt_element is None:
            t_dt2E_dt.append(None)
            continue
        t_dt2E_dt_element = time * dt2E_dt_element
        t_dt2E_dt_element.gamma_method()
        t_dt2E_dt.append(t_dt2E_dt_element)

        upper_bound = t_dt2E_dt_element.value + t_dt2E_dt_element.dvalue
        lower_bound = t_dt2E_dt_element.value - t_dt2E_dt_element.dvalue
        if lower_bound < threshold < upper_bound:
            indices_to_fit.append(index)
        elif indices_to_fit:
            # Have left window; stop to avoid having two separate ranges
            break
        elif lower_bound > threshold:
            # Missed window entirely
            indices_to_fit = list(range(index - 3, index + 2))
    else:
        # Flow doesn't extend far enough, so don't get a result
        raise UnreachableThreshold("No window found")

    t_dt2E_dt_corr = pe.Corr(t_dt2E_dt)
    t_dt2E_dt_corr.gamma_method()
    fit_result = t_dt2E_dt_corr.fit(
        quadratic_fit_form,
        fitrange=[min(indices_to_fit), max(indices_to_fit)],
        silent=True,
    )
    a, b, c = fit_result.fit_parameters
    solution_time_offset = get_quadratic_left_solution(a, b, c - threshold)
    w0 = (solution_time_offset * (times[1] - times[0])) ** 0.5
    return w0, t_dt2E_dt_corr


def plot_flow(
    plaquette_flow,
    clover_flow,
    metadata,
    threshold,
    time_step,
    plaquette_w0=None,
    clover_w0=None,
):
    fig, ax = plt.subplots(layout="constrained")

    ax.set_xlabel("$t / a$")
    ax.set_ylabel(r"$t\frac{\mathrm{d}\langle{t^2 E\rangle}}{\mathrm{d}t}$")

    for colour_idx, (flow, label, marker, w0) in enumerate(
        [
            (plaquette_flow, "Plaquette", "o", plaquette_w0),
            (clover_flow, "Clover", "s", clover_w0),
        ]
    ):
        plot_x, plot_y, plot_yerr = flow.plottable()
        ax.errorbar(
            np.array(plot_x) * time_step,
            plot_y,
            yerr=plot_yerr,
            ls="none",
            label=label,
            marker=marker,
            color=f"C{colour_idx}",
        )
        if w0 is not None:
            t_w0 = w0**2
            t_w0.gamma_method()
            ax.axvline(t_w0.value, color=f"C{colour_idx}", dashes=(2, 3))
            ax.axvspan(
                t_w0.value - t_w0.dvalue,
                t_w0.value + t_w0.dvalue,
                color=f"C{colour_idx}",
                alpha=0.2,
            )

    ax.legend(loc="best")
    ax.set_title(
        (
            "${nt}\\times{nx}\\times{ny}\\times{nz}$, $\\beta={beta}$, $\\mathcal{{W}}_0={threshold}$"
        ).format(**metadata, threshold=threshold)
    )
    ax.axhline(threshold, color="black", dashes=(2, 5))

    return fig


def main():
    """
    Compute the flow scale w0 and output
    """
    args = get_args()
    metadata, data = read_flows(args.wf_log)
    time_step = get_time_step(data["flow_times"])
    ensemble_name = name_ensemble(metadata)

    if args.W0 is None:
        # TODO: SU(3) value; scale this based on Casimir of group.
        w0_threshold = 0.3
    else:
        w0_threshold = args.W0

    try:
        plaquette_w0, plaquette_flow = compute_w0(
            data["flow_times"],
            data["plaquette_energy_densities"],
            ensemble_name,
            data["trajectory"],
            w0_threshold,
        )
    except UnreachableThreshold:
        plaquette_w0 = None
        plaquette_flow = None
    try:
        clover_w0, clover_flow = compute_w0(
            data["flow_times"],
            data["clover_energy_densities"],
            ensemble_name,
            data["trajectory"],
            w0_threshold,
        )
    except UnreachableThreshold:
        clover_w0 = None
        clover_flow = None

    if args.output_filename is None:
        print(f"plaquette_w0: {plaquette_w0}, clover_w0: {clover_w0}")
    else:
        pe.input.json.dump_dict_to_json(
            {
                "plaquette_w0": plaquette_w0,
                "clover_w0": clover_w0,
            },
            args.output_filename,
            description=metadata,
        )

    if args.plot_filename:
        plt.style.use(args.plot_styles)
        plot_flow(
            plaquette_flow,
            clover_flow,
            metadata,
            w0_threshold,
            time_step,
            plaquette_w0,
            clover_w0,
        ).savefig(args.plot_filename)


if __name__ == "__main__":
    main()
