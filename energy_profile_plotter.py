import os
import argparse
import matplotlib.pyplot as plt
from glob import glob

COEF_DALTON_TO_EV = 27.2114


def plot_energy_profile(energies: list[list], points_x: list[float], line_1_style: str, line_2_style: str,
                        line_1_width: float, line_2_width: float, line_1_color: str, line_2_color: str,
                        line_1_marker: str, line_2_marker: str, line_1_marker_size: int, line_2_marker_size: int,
                        axis_line_width: float, label_font_size: float, x_min: float, y_min: float, x_max: float,
                        y_max: float, plot_title: str, fig=None, canvas=None):
    """
    Read input files from folder, calculate ereorg, mingap and DeltaG and build an energies plot on both states
    :param energies: list of energies. Each element is a list of two float energy values in two states
    :param points_x: list of x coordinates for each point, represented by each file
    :param line_1_style: line style for state_1 graph
    :param line_2_style: line style for state_2 graph
    :param line_1_width: line width for state_1 graph
    :param line_2_width: line width for state_2 graph
    :param line_1_color: line color for state_1 graph
    :param line_2_color: line color for state_2 graph
    :param line_1_marker: line marker for state_1 graph
    :param line_2_marker: line marker for state_2 graph
    :param line_1_marker_size: line marker size for state_1 graph
    :param line_2_marker_size: line marker size for state_2 graph
    :param axis_line_width:
    :param label_font_size:
    :param x_min: min x coordinate on a graph
    :param y_min: min y coordinate on a graph
    :param x_max: max x coordinate on a graph
    :param y_max: max y coordinate on a graph
    :param plot_title:
    :param fig: if set - draw the plot there
    :param canvas: if set - draw the plot there
    :return: ereorg, mingap, delta_g (float, float, float)
    """

    if not fig:
        fig = plt.figure(num=None, figsize=(10, 8), facecolor='white', edgecolor='white', frameon=True, clear=True)
    else:
        fig.clear()
    plt.title(plot_title, fontsize='xx-large')
    ax = fig.gca()
    ax.set_xlabel('Reaction coordinate', fontsize='xx-large')
    ax.set_ylabel('Energy (eV)', fontsize='xx-large')
    ax.tick_params(labelsize=label_font_size)
    plt.axis((x_min, x_max, y_min, y_max))
    plt.setp(ax.spines.values(), linewidth=axis_line_width)

    line1 = ax.plot(points_x, [e[0] for e in energies])
    plt.setp(line1, marker=line_1_marker, markersize=line_1_marker_size, color=line_1_color, linewidth=line_1_width,
             linestyle=line_1_style, label='State 1')
    line2 = ax.plot(points_x, [e[1] for e in energies])
    plt.setp(line2, marker=line_2_marker, markersize=line_2_marker_size, color=line_2_color, linewidth=line_2_width,
             linestyle=line_2_style, label='State 2')

    ax.legend(loc='center right', fontsize='large')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if canvas:
        canvas.draw()
    else:
        plt.show()


def calculate_profile(energies: list[list], line_order: list[int]):
    """
    Calculate mingap, ereorg and delta_g based on energy values in a list of states.
    :param energies: list of energies. Each element is a list of two float energy values in two states
    :param line_order: determine, which state line lies lower, so we can determine DeltaG from it
    :return: ereorg, mingap, delta_g (float, float, float)
    """

    mingap = float('inf')
    ereorg = 0
    delta_g = float('inf')

    for i, energy in enumerate(energies):
        # Calculate gap between higher and lower line state
        gap = energy[line_order[1]] - energy[line_order[0]]

        # Calculate minimum gap
        mingap = gap if gap < mingap else mingap
        if energy[line_order[0]] * energy[line_order[1]] == 0 and gap > ereorg:
            # Determine ereorg: gap in a point, where state energy is 0
            ereorg = gap
        if i == 0 or i == len(energies) - 1:
            continue
        if energies[i - 1][line_order[0]] > energy[line_order[0]] > 0 and \
                energies[i + 1][line_order[0]] > energy[line_order[0]] and \
                energy[line_order[0]] < delta_g:
            # DeltaG - smallest local non-zero minimum of an energy
            delta_g = energy[line_order[0]]

    if delta_g == float('inf'):
        border_values = [energies[i][line_order[0]] for i in (0, -1) if energies[i][line_order[0]] > 0]
        if border_values:
            # In case if there is no local non-zero minimums - take the smallest non-zero border value as DeltaG
            delta_g = min(border_values)
        else:
            # In case if there is no local non-zero minimums and non-zero border values DeltaG = 0
            delta_g = 0

    return ereorg, mingap, delta_g


def energy_profile(folder_path: str, line_1_style: str, line_2_style: str, line_1_width: float, line_2_width: float,
                   line_1_color: str, line_2_color: str, line_1_marker: str, line_2_marker: str,
                   line_1_marker_size: int, line_2_marker_size: int, axis_line_width: float, label_font_size: float,
                   x_min: float, y_min: float, x_max: float, y_max: float, plot_title: str, fig=None, canvas=None):
    """
    Read input files from folder, calculate ereorg, mingap and DeltaG and build an energies plot on both states
    :param folder_path: folder with .out files
    :param line_1_style: line style for state_1 graph
    :param line_2_style: line style for state_2 graph
    :param line_1_width: line width for state_1 graph
    :param line_2_width: line width for state_2 graph
    :param line_1_color: line color for state_1 graph
    :param line_2_color: line color for state_2 graph
    :param line_1_marker: line marker for state_1 graph
    :param line_2_marker: line marker for state_2 graph
    :param line_1_marker_size: line marker size for state_1 graph
    :param line_2_marker_size: line marker size for state_2 graph
    :param axis_line_width:
    :param label_font_size:
    :param x_min: min x coordinate on a graph
    :param y_min: min y coordinate on a graph
    :param x_max: max x coordinate on a graph
    :param y_max: max y coordinate on a graph
    :param plot_title:
    :param fig: if set - draw the plot there
    :param canvas: if set - draw the plot there
    :return: ereorg, mingap, delta_g (float, float, float)
    """

    files_list = glob(os.path.join(folder_path, '*.out'))
    assert files_list, "Folder doesn't contain any .out files"

    # Get x-coordinate for all energy values
    points_x = [filepath.rsplit('.', maxsplit=2)[-2] for filepath in files_list]
    points_x = [int(x) / 10 ** max(len(p) - 1 for p in points_x) for x in points_x]

    energies = []
    min_energy = float('inf')   # Min state energy will be considered as zero

    lower_line = 0  # 0 if state_1 line is lower, 1 - if state_2

    # Read input data
    for filepath in files_list:
        with open(filepath, encoding='utf8') as f:
            for line in f:
                if '*** XMC-QDPT2 ENERGIES ***' in line:
                    for _ in range(2):
                        f.readline()
                    state_1 = float(f.readline().strip().split()[-1])
                    state_2 = float(f.readline().strip().split()[-1])

                    lower_line = int(state_2 < state_1)  # Determine, if state_2 is lower than state_1

                    if [state_1, state_2][lower_line] < min_energy:
                        min_energy = [state_1, state_2][lower_line]

                    energies.append([state_1, state_2])
                    break

    line_order = [0, 1] if lower_line == 0 else [1, 0]

    # Recalculate state energies from Daltons to Ev
    energies = [[(e - min_energy) * COEF_DALTON_TO_EV for e in state] for state in energies]

    ereorg, mingap, delta_g = calculate_profile(energies, line_order)

    plot_energy_profile(energies, points_x, line_1_style, line_2_style, line_1_width, line_2_width, line_1_color,
                        line_2_color, line_1_marker, line_2_marker, line_1_marker_size, line_2_marker_size,
                        axis_line_width, label_font_size, x_min, y_min, x_max, y_max, plot_title, fig, canvas)

    return ereorg, mingap, delta_g


def main(folder_path: str, line_1_style: str, line_2_style: str, line_1_width: float, line_2_width: float,
         line_1_color: str, line_2_color: str, line_1_marker: str, line_2_marker: str, line_1_marker_size: int,
         line_2_marker_size: int, axis_line_width: float, label_font_size: float, x_min: float, y_min: float,
         x_max: float, y_max: float, plot_title: str):
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a correct folder path")

    results = energy_profile(folder_path, line_1_style, line_2_style, line_1_width, line_2_width, line_1_color,
                             line_2_color, line_1_marker, line_2_marker, line_1_marker_size, line_2_marker_size,
                             axis_line_width, label_font_size, x_min, y_min, x_max, y_max, plot_title)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path')
    parser.add_argument('--line_1_style', choices=['-', '--', '-.', '..'], default='-')
    parser.add_argument('--line_2_style', choices=['-', '--', '-.', '..'], default='-')
    parser.add_argument('--line_1_width', type=float, default=1.0)
    parser.add_argument('--line_2_width', type=float, default=1.0)
    parser.add_argument('--line_1_color',
                        choices=["orange", "blue", "green", "red", "cyan", "magenta", "yellow", "black"],
                        default='blue')
    parser.add_argument('--line_2_color',
                        choices=["orange", "blue", "green", "red", "cyan", "magenta", "yellow", "black"],
                        default='orange')
    parser.add_argument('--line_1_marker', choices=[".", "o", "v", "^", "s", "+", "x", "D", "d", "_"], default='.')
    parser.add_argument('--line_2_marker', choices=[".", "o", "v", "^", "s", "+", "x", "D", "d", "_"], default='.')
    parser.add_argument('--line_1_marker_size', choices=list(range(11)), default=2)
    parser.add_argument('--line_2_marker_size', choices=list(range(11)), default=2)
    parser.add_argument('--axis_line_width', type=float, default=1.0)
    parser.add_argument('--label_font_size', type=float, default=10.0)
    parser.add_argument('--x_min', type=float, default=0.0)
    parser.add_argument('--y_min', type=float, default=0.0)
    parser.add_argument('--x_max', type=float, default=1.0)
    parser.add_argument('--y_max', type=float, default=2.0)
    parser.add_argument('--plot_title', default='Energy profile')

    args = parser.parse_args()
    main(**vars(args))
