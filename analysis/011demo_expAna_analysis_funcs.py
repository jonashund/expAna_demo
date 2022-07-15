import expAna
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from matplotlib.legend_handler import HandlerTuple

from expAna.misc import InputError


def plt_style(font="utopia"):
    # Custom matplotlib style for PhD thesis
    # Custom matplotlib styles are usually located in `print(matplotlib.get_configdir())`
    plt.style.use("./phd.mplstyle")

    # Set rcParams for LaTeX
    latex_preamble_dict = {
        "palatino": r"\usepackage[T1]{fontenc} \usepackage{mathtools} \usepackage{xfrac} \usepackage{newpxtext,newpxmath} \newcommand*{\units}[1]{\mbox{[\ifx&#1&$\hbox{-}$\else#1\fi]}}",
        "computer_modern": r"\usepackage[T1]{fontenc} \usepackage{mathtools} \usepackage{xfrac} \newcommand*{\units}[1]{\mbox{[\ifx&#1&$\hbox{-}$\else#1\fi]}}",
        "libertine": r"\usepackage{mathtools} \usepackage{xfrac} \usepackage{libertine} \usepackage[libertine]{newtxmath} \usepackage[T1]{fontenc} \newcommand*{\units}[1]{\mbox{[\ifx&#1&$\hbox{-}$\else#1\fi]}}",
        "utopia": r"\usepackage{mathtools} \usepackage{xfrac} \usepackage{libertine} \usepackage[utopia]{newtxmath} \usepackage[T1]{fontenc} \newcommand*{\units}[1]{\mbox{[\ifx&#1&$\hbox{-}$\else#1\fi]}}",
        "beamer": r"\usepackage{mathtools} \usepackage{xfrac} \usepackage{helvet} \usepackage[utopia]{newtxmath} \usepackage[T1]{fontenc} \newcommand*{\units}[1]{\mbox{[\ifx&#1&$\hbox{-}$\else#1\fi]}}",
    }
    phd_rcParams = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": latex_preamble_dict[font],
        # Use xxpt font in plots, to match xxpt font in document
        "font.size": 12,
        "axes.labelsize": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
    matplotlib.rcParams.update(phd_rcParams)


def analysis_stress(
    compare, select=None, experiment_list=None, ignore_list=None,
):
    analysis = expAna.analysis.Analysis(type="stress")

    analysis.setup(
        compare=compare,
        select=select,
        experiment_list=experiment_list,
        ignore_list=ignore_list,
    )
    ####################################################################################
    # COMPUTE AVERAGE CURVES, ETC.
    ####################################################################################
    analysis.compute_data_stress()

    return analysis


def analysis_vol_strain(
    compare, select=None, experiment_list=None, ignore_list=None,
):
    analysis = expAna.analysis.Analysis(type="vol_strain")
    analysis.setup(
        compare=compare,
        select=select,
        experiment_list=experiment_list,
        ignore_list=ignore_list,
    )
    ####################################################################################
    # COMPUTE AVERAGE CURVES, ETC.
    ####################################################################################
    analysis.compute_data_vol_strain()

    return analysis


def analysis_force(
    compare,
    select=None,
    experiment_list=None,
    ignore_list=None,
    x_lim=None,
    displ_shift=None,
):

    work_dir = os.getcwd()
    instron_data_dir = os.path.join(work_dir, "data_instron")

    analysis = expAna.analysis.Analysis(type="force")
    analysis.setup(
        exp_data_dir=instron_data_dir,
        compare=compare,
        select=select,
        experiment_list=experiment_list,
        ignore_list=ignore_list,
    )
    ####################################################################################
    # COMPUTE AVERAGE CURVES, ETC.
    ####################################################################################
    analysis.compute_data_force(displ_shift)

    return analysis


def analysis_poissons_ratio(
    compare, select=None, experiment_list=None, ignore_list=None,
):

    os.makedirs(vis_export_dir, exist_ok=True)

    analysis = expAna.analysis.Analysis(type="poissons_ratio")
    analysis.setup(
        compare=compare,
        select=select,
        experiment_list=experiment_list,
        ignore_list=ignore_list,
    )
    ####################################################################################
    # COMPUTE AVERAGE CURVES, ETC.
    ####################################################################################
    analysis.compute_data_poissons_ratio()

    return analysis


def get_analysis_func(analysis_type):
    analysis_func = {
        "stress": analysis_stress,
        "force": analysis_force,
        "vol_strain": analysis_vol_strain,
        "poissons_ratio": analysis_poissons_ratio,
    }
    return analysis_func[analysis_type]


def get_analysis_type_props(type):
    analysis_type_props = {
        "force": {
            "x_lim": 3.75,
            "y_lim": 3.0,
            "x_label": r"Displacement $u$ \units{mm}",
            "y_label": r"Force $F$ \units{kN}",
        },
        "stress": {
            "x_lim": 0.9,
            "y_lim": 100,
            "x_label": r"Log. strain $\varepsilon$ \units{}",
            "y_label": r"True stress $\sigma$ \units{MPa}",
        },
        "vol_strain": {
            "x_lim": 0.9,
            "y_lim": 0.4,
            "x_label": r"Log. strain $\varepsilon$ \units{}",
            "y_label": r"Volume strain $\varepsilon_\mathsf{v}$ \units{}",
        },
        "poissons_ratio": {
            "x_lim": 0.9,
            "y_lim": 0.5,
            "x_label": r"Log. strain $\varepsilon$ \units{}",
            "y_label": r"Poisson's ratio $\nu$ \units{}",
        },
    }
    return analysis_type_props[type]


def get_labels_and_titles(analysis, material, experiment_type):
    legend_dict = {
        "sent": {
            "specimen_orientation": {
                "title": r"Specimen orientation:",
                "handles": [r"Parallel to flow", "Perpendicular to flow"],
            },
            "crosshead_speed": {
                "title": r"Crosshead speed:",
                "handles": [r"$\dot{u}=0.6$\,mm/s", r"$\dot{u}=6.0$\,mm/s"],
            },
        },
        "uniax_tension": {
            "specimen_orientation": {
                "title": r"Specimen orientation:",
                "handles": [r"Parallel to flow", "Perpendicular to flow"],
            },
            "crosshead_speed": {
                "title": r"Strain rate:",
                "handles": [
                    r"$\dot{\varepsilon} = 0.01$\,s$^{-1}$",
                    r"$\dot{\varepsilon} = 0.1$\,s$^{-1}$",
                ],
            },
        },
    }
    title_dict_a = {
        "4555": r"PC/ABS~(45/55)",
        "6040": r"PC/ABS~(60/40)",
        "7030": r"PC/ABS~(70/30)",
    }
    title_dict_b = {
        "sent": {
            "specimen_orientation": {
                "parallel_to_flow": r"specimen orientation: parallel to flow",
                "perpendicular_to_flow": r"specimen orientation: perpendicular to flow",
            },
            "crosshead_speed": {
                "0.6": r"crosshead speed: $\dot{u}=0.6$\,mm/s",
                "6": r"crosshead speed: $\dot{u}=6.0$\,mm/s",
            },
        },
        "uniax_tension": {
            "specimen_orientation": {
                "parallel_to_flow": r"specimen orientation: parallel to flow",
                "perpendicular_to_flow": r"specimen orientation: perpendicular to flow",
            },
            "crosshead_speed": {
                "0.1": r"strain rate: $\dot{\varepsilon} = 0.01$\,s$^{-1}$",
                "1": r"strain rate: $\dot{\varepsilon} = 0.1$\,s$^{-1}$",
            },
        },
    }

    return {
        "title": title_dict_a[material]
        + ",\, "
        + title_dict_b[experiment_type][analysis.select_key][analysis.select_value],
        "legend": legend_dict[experiment_type][analysis.compare_key],
    }


def create_analysis_figure(width=5, height=3.75, constrained_layout=False):
    plt_style()
    fig_1 = plt.figure(
        figsize=(1.5 * width, height), constrained_layout=constrained_layout
    )
    axes_1 = plt.subplot(111)

    axes_1.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axes_1.yaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axes_1.grid(color="lightgrey", linewidth=0.33, linestyle="-")

    axes_1.tick_params(direction="out", pad=5)
    axes_1.tick_params(bottom=True, left=True, top=False, right=False)

    return fig_1, axes_1


def add_mean_to_plot(axes, x, y, color, label=None):
    axes.plot(
        x, y, label=label, linewidth=2, zorder=10, color=color,
    )


def add_raw_data_to_plot(axes, xs, ys, color, label=None):
    for i in range(len(xs)):
        if i == 0:
            axes.plot(
                xs[i],
                ys[i],
                linewidth=0.75,
                linestyle="--",
                dashes=(7, 7),
                zorder=1,
                color=color,
                label=None,
                # alpha=0.33,
            )
        else:
            axes.plot(
                xs[i],
                ys[i],
                linewidth=0.75,
                linestyle="--",
                dashes=(7, 7),
                zorder=1,
                color=color,
                # alpha=0.33,
            )


def add_confidence_to_plot(axes, x_mean, y_mean, error, color, label=None):
    axes.fill_between(
        x_mean, y_mean - error, y_mean + error, alpha=0.2, facecolor=color, label=label
    )


def add_annotations(fig, axes, analysis, annotate_dict, compare_value=None, color=None):
    if color is None:
        color = "black"
    else:
        pass

    if compare_value is None:
        compare_values = analysis.compare_values
    else:
        compare_values = [compare_value]

    for compare_value in compare_values:
        for experiment_name in np.array(
            analysis.dict[compare_value]["experiment_list"], dtype=object
        )[analysis.dict[compare_value]["y_indices"]]:
            if experiment_name in annotate_dict:

                if "arrowprops" not in annotate_dict[experiment_name]:
                    arrowprops = dict(
                        arrowstyle="-|>",
                        color=color,
                        shrinkA=5,
                        shrinkB=5,
                        patchA=None,
                        patchB=None,
                        connectionstyle="angle,angleA=0,angleB=90,rad=0",
                    )
                else:
                    arrowprops = annotate_dict[experiment_name]["arrowprops"]

                axes.annotate(
                    annotate_dict[experiment_name]["text"],
                    xy=annotate_dict[experiment_name]["xy"],
                    xytext=annotate_dict[experiment_name]["xytext"],
                    arrowprops=arrowprops,
                    color=color,
                    fontsize=10,
                    horizontalalignment="left",
                    verticalalignment="center",
                )
    return


def fill_axes(axes, analysis, color=None, mean_only=False):
    handles = []

    for compare_value in analysis.compare_values:
        # set current color
        if color is None:
            current_color = next(axes._get_lines.prop_cycler)["color"]
        else:
            current_color = color
        # define data to plot
        if analysis.type == "force":
            x_mean = analysis.dict[compare_value]["x_mean"]
            x_mean_max = x_mean.max()
            idx_max = np.argwhere(x_mean < x_mean_max - 1.0e-1).max()
            x_mean = x_mean[:idx_max]
            y_mean = analysis.dict[compare_value]["y_mean"][:idx_max]
            error = 1.959 * analysis.dict[compare_value]["y_sem"][:idx_max]
        else:
            x_mean = analysis.dict[compare_value]["x_mean"]
            y_mean = analysis.dict[compare_value]["y_mean"]
            error = 1.959 * analysis.dict[compare_value]["y_sem"]
            if analysis.type == "poissons_ratio":
                start_idx = np.argmax(x_mean > 0.03)
                x_mean = x_mean[start_idx:]
                y_mean = y_mean[start_idx:]
                error = error[start_idx:]
            else:
                pass

        xs = np.array(analysis.dict[compare_value]["xs"], dtype=object)[
            analysis.dict[compare_value]["y_indices"]
        ]
        ys = np.array(analysis.dict[compare_value]["ys"], dtype=object)[
            analysis.dict[compare_value]["y_indices"]
        ]

        # add plots analysis mean
        add_mean_to_plot(axes, x_mean, y_mean, current_color, label="Mean")

        if mean_only == False:
            add_raw_data_to_plot(axes, xs, ys, color=current_color, label="Raw data")

            add_confidence_to_plot(
                axes, x_mean, y_mean, error, color=current_color, label="95% CI"
            )
        else:
            pass

        dummy_plot = axes.fill_between(
            [-1, -0.5],
            [-0.99, -0.99],
            [-1.01, -1.01],
            zorder=1,
            facecolor=current_color,
            label=r"color_sample",
        )
        handles.append(dummy_plot)
    return handles


def fill_analysis_figure(experiment_type, analysis, legend_title, legend_labels):
    # analysis type dict
    props_dict = get_analysis_type_props(analysis.type)

    if analysis.type == "force" and experiment_type == "uniax_tension":
        props_dict["x_lim"] = 35.0
        props_dict["y_lim"] = 2.5
    else:
        pass

    fig_1, axes_1 = create_analysis_figure()

    axes_1.set_xlabel(props_dict["x_label"])
    axes_1.set_ylabel(props_dict["y_label"])
    axes_1.set_xlim(0, props_dict["x_lim"])
    axes_1.set_ylim(0, props_dict["y_lim"])

    handles = fill_axes(axes_1, analysis)

    # resize axis on figure canvas
    fig_1.tight_layout()
    axes_1_box = axes_1.get_position()
    axes_1.set_position(
        [axes_1_box.x0, axes_1_box.y0, axes_1_box.width * 0.67, axes_1_box.height]
    )
    # Create a legend to indicate the types of data visualised
    (dummy_mean,) = axes_1.plot(
        0, 0, linewidth=2, zorder=1, color="black", label=r"Mean"
    )
    (dummy_raw_data,) = axes_1.plot(
        0,
        0,
        zorder=1,
        color="black",
        linewidth=0.75,
        linestyle="--",
        dashes=(7, 7),
        label=r"Raw data",
    )
    dummy_ci = axes_1.fill_between(
        [-1, -0.5],
        [-0.99, -0.99],
        [-1.01, -1.01],
        zorder=1,
        alpha=0.2,
        facecolor="black",
        label=r"95\,\% CI",
    )

    key_to_legend = plt.legend(
        handles=[dummy_mean, dummy_raw_data, dummy_ci],
        loc="upper left",
        bbox_to_anchor=(1, 0.475),
        frameon=False,
    )

    # Add the legend manually to the current Axes.
    plt.gca().add_artist(key_to_legend)

    axes_1.legend(
        loc="lower left",
        bbox_to_anchor=(1, 0.54),
        title=legend_title,
        handles=[(handles[0]), (handles[1]),],  # The line objects
        labels=[legend_labels[0], legend_labels[1]],  # The labels for each line
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
    )

    return fig_1, axes_1


def export_current_figure(analysis, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        os.path.join(out_dir, f"{analysis.export_prefix}_comparison.pdf",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.savefig(
        os.path.join(out_dir, f"{analysis.export_prefix}_comparison.png",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()


def make_analysis_figure(
    select,
    compare,
    material,
    experiment_type,
    analysis_type,
    ignore_list=None,
    experiment_list=None,
    annotate_dict=None,
):
    analysis = get_analysis_func(analysis_type)(
        select=select,
        compare=compare,
        ignore_list=ignore_list,
        experiment_list=experiment_list,
    )

    labels_dict = get_labels_and_titles(analysis, material, experiment_type)

    fig_1, axes_1 = fill_analysis_figure(
        experiment_type=experiment_type,
        analysis=analysis,
        legend_title=labels_dict["legend"]["title"],
        legend_labels=labels_dict["legend"]["handles"],
    )

    if annotate_dict is not None:
        add_annotations(fig_1, axes_1, analysis, annotate_dict)
    else:
        pass

    axes_1.set_title(labels_dict["title"],)

    if analysis.compare_key == "specimen_orientation":
        line_coords = [(0.75, 0.55), (0.95, 0.55)]
    elif analysis.compare_key == "crosshead_speed":
        line_coords = [(0.76, 0.55), (0.92, 0.55)]

    axes_1.annotate(
        "",
        xy=line_coords[0],
        xycoords="figure fraction",
        xytext=line_coords[1],
        textcoords="figure fraction",
        ha="right",
        va="top",
        arrowprops=dict(arrowstyle="-"),
    )
    export_current_figure(analysis, os.path.join(os.getcwd(), "expAna_plots"))

    expAna.data_trans.export_analysis(
        analysis, out_filename=f"analysis_{analysis.export_prefix}.pickle",
    )

