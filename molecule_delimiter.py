import argparse
import os
import numpy as np
import networkx as nx


class Atom:
    """
    Dataclass containing information about atom (one line from .xyz file)
    attributes:
        line:
            original line content
        coord:
            atom coordinates
    """

    def __init__(self, line: str):
        self.line = line.strip()
        self.coord = np.array([float(i) for i in line.split(maxsplit=1)[1].split()])


def build_cartesian_graph(input_file: str, max_bond_length: float) -> nx.Graph:
    """
    Read .xyz file and build connectivity graph on the condition: if distance between two atoms is <= max_bond_length
    they are considered as connected
    :param input_file:
        Input .xyz file containing atoms information
    :param max_bond_length:
        Maximum distance possible between two atoms to be considered as connected
    :return: graph (networkx.Graph) of all atoms and their connections
    """

    with open(input_file) as f:
        cartesian = [Atom(line) for line in f.readlines()[2:]]

    cartesian_graph = nx.Graph()

    for atom in cartesian:
        cartesian_graph.add_node(atom)

    for i in range(len(cartesian)):
        for j in range(i + 1, len(cartesian)):
            if np.linalg.norm(cartesian[i].coord - cartesian[j].coord) <= max_bond_length:
                cartesian_graph.add_edge(cartesian[i], cartesian[j])

    return cartesian_graph


def molecules_delimiter(input_file: str, min_atoms_per_molecule: int, max_bond_length: float, output_folder: str):
    """
    Split molecule into monomers by connecting atoms based on their distance from each other and extract connectivity \
    components from resulting graph
    :param input_file:
        Input .xyz file containing atoms information
    :param min_atoms_per_molecule:
        If after splitting a connectivity component consists of less than min_atoms_per_molecule atoms - don't print
        it to file
    :param max_bond_length:
        Maximum distance possible between two atoms to be considered as connected
    :param output_folder:
        Folder to put result to
    :return:
    """
    cartesian_graph = build_cartesian_graph(input_file, max_bond_length)
    molecules = list(
        molecule for molecule in nx.connected_components(cartesian_graph) if len(molecule) >= min_atoms_per_molecule)
    for idx, molecule in enumerate(molecules):
        out_filepath = os.path.join(
            output_folder,
            os.path.splitext(os.path.basename(input_file))[0] + f'_pair{str(idx).zfill(len(str(idx)))}.xyz')
        with open(out_filepath, 'w') as f:
            f.writelines(f"{len(molecule)}\n{os.path.splitext(os.path.basename(out_filepath))[0]}\n")
            f.writelines(line.line + '\n' for line in molecule)


def main(input_file: str, min_atoms_per_molecule: int, max_bond_length: float, output_folder: str):
    if not os.path.isdir(output_folder):
        raise OSError(f'{output_folder} is not a folder. Please provide a correct folder path')
    molecules_delimiter(input_file, min_atoms_per_molecule, max_bond_length, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='Input .xyz cartesian file')
    parser.add_argument('--min_atoms_per_molecule', type=int)
    parser.add_argument('--max_bond_length', type=float)
    parser.add_argument('--output_folder')
    args = parser.parse_args()
    main(**vars(args))
