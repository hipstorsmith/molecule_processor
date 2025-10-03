import abc
import argparse
import os
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree, distance_matrix
from collections import Counter


class Atom:
    """
    Dataclass containing information about atom (one line from .xyz file)
    attributes:
        line:
            original line content
        coord:
            atom coordinates
    """

    def __init__(self, line: str, idx: int = 0):
        self.line = line.strip()
        self.idx = idx
        self.name, coord = line.split(maxsplit=1)
        self.coord = np.array([float(i) for i in coord.split()])


class Molecule(abc.ABC):
    """
    Abstract dataclass representing a molecule for inheritance
    attributes:
        atoms:
            list of Atom objects
    """

    def __init__(self):
        self.atoms = set()


class Monomer(Molecule):
    """
    Dataclass representing a monomer
    attributes:
        atoms:
            list of Atom objects
    """

    def __init__(self, molecule: set[Atom]):
        super().__init__()
        self.atoms = sorted(molecule, key=lambda a: a.idx)


class Dimer(Molecule):
    """
    Dataclass representing a dimer
    attributes:
        atoms:
            list of Atom objects
        type_counts:
            number of atoms of each type
        distance_signature:
            pairwise distances between atoms
        centered_coords:
            list of translated to (0, 0, 0) centroid atom coordinates
    """

    def __init__(self, monomer_1: Monomer, monomer_2: Monomer):
        super().__init__()
        self.atoms = sorted(monomer_1.atoms + monomer_2.atoms, key=lambda a: a.idx)
        self.type_counts = Counter(atom.name for atom in self.atoms)
        self.distance_signature = np.sort(
            np.sort(distance_matrix(self.centered_coords, self.centered_coords), axis=1), axis=0)

    @property
    def centered_coords(self) -> np.ndarray:
        """
        List of translated to (0, 0, 0) centroid atom coordinates
        """

        coord = np.vstack([atom.coord for atom in self.atoms])
        centroid = np.mean(coord, axis=0)
        return coord - centroid


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
        cartesian = [Atom(line, i) for i, line in enumerate(f.readlines()[2:])]
    coords = np.vstack([a.coord for a in cartesian])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=max_bond_length, output_type='set')
    cartesian_graph = nx.Graph()

    for atom in cartesian:
        cartesian_graph.add_node(atom)

    for i, j in pairs:
        cartesian_graph.add_edge(cartesian[i], cartesian[j])

    return cartesian_graph


def split_to_monomers(input_file: str, min_atoms_per_molecule: int, max_bond_length: float) -> list[Monomer]:
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

    :return: list of Monomers
    """

    cartesian_graph = build_cartesian_graph(input_file, max_bond_length)
    molecules = list(
        Monomer(molecule) for molecule in nx.connected_components(cartesian_graph) if
        len(molecule) >= min_atoms_per_molecule)

    return molecules


def monomers_to_dimers(monomers: list[Monomer], contact_distance: float) -> list[Dimer]:
    """
    Merge monomers into dimers. If in a pair of monomers MA and MB there is at least one pair of atoms a (of MA) and b
    (of MB) with distance <= contact_distance - add these molecules into dimer
    :param monomers:
        list of monomers
    :param contact_distance:
        maximum distance between two atoms in monomers to be considered as connected
    :return:
    """

    if len(monomers) < 2:
        return []

    monomer_coords = []
    atom_monomer_map = []
    monomer_pair_idxs = set()

    for i, monomer in enumerate(monomers):
        for atom in monomer.atoms:
            monomer_coords.append(atom.coord)
            atom_monomer_map.append(i)

    monomer_coords = np.vstack(monomer_coords)
    tree = cKDTree(monomer_coords)
    # Find all atoms with distance < contact_distance
    atom_pairs = tree.query_pairs(r=contact_distance, output_type='set')
    for i, j in atom_pairs:
        monomer_i_idx = atom_monomer_map[i]
        monomer_j_idx = atom_monomer_map[j]
        if monomer_i_idx != monomer_j_idx:
            # Don't add twin pairs
            monomer_pair_idxs.add(tuple(sorted([monomer_i_idx, monomer_j_idx])))

    return [Dimer(monomers[i], monomers[j]) for (i, j) in monomer_pair_idxs]


def deduplicate_dimers(dimers: list[Dimer], eps=10 ** -6) -> list[Dimer]:
    """
    Remove isomorphic dimers
    :param dimers:
        list of dimers to clean-up
    :param eps:
        tolerance to avoid float errors
    :return:
        list of dimers without isomorphic ones
    """

    unique_dimers: list[Dimer] = []

    for candidate in dimers:
        is_duplicate = False
        for dimer in unique_dimers:
            # Checking amounts of atoms of different types. Not matching - not a duplicate
            if candidate.type_counts != dimer.type_counts:
                continue
            # If amount of atoms are matching - total amount of atoms is equal. Pairwise distances can be compared
            # Checking if matrices of pairwise distances are close. If they are not - not a duplicate

            if not np.allclose(candidate.distance_signature, dimer.distance_signature, atol=eps):
                continue
            is_duplicate = True

        if not is_duplicate:
            unique_dimers.append(candidate)

    return unique_dimers


def output_molecules(molecules: list[Molecule], input_file: str, output_folder: str):
    """
    :param molecules:
        list of Monomers or Dimers
    :param input_file:
        input xyz file
    :param output_folder:
        Folder to put result into
    """

    for idx, molecule in enumerate(molecules):
        out_filepath = os.path.join(
            output_folder,
            os.path.splitext(os.path.basename(input_file))[0] + f'_pair{str(idx).zfill(len(str(idx)))}.xyz')
        with open(out_filepath, 'w') as f:
            f.writelines(f"{len(molecule.atoms)}\n{os.path.splitext(os.path.basename(out_filepath))[0]}\n")
            f.writelines(line.line + '\n' for line in molecule.atoms)


def molecules_delimiter(input_file: str, min_atoms_per_molecule: int, max_bond_length: float, output_folder: str,
                        contact_distance: float, merge_to_dimers: bool, drop_duplicates: bool):
    molecules = split_to_monomers(input_file, min_atoms_per_molecule, max_bond_length)
    if merge_to_dimers:
        molecules = monomers_to_dimers(molecules, contact_distance)
        if drop_duplicates:
            molecules = deduplicate_dimers(molecules)

    output_molecules(molecules, input_file, output_folder)


def main(input_file: str, min_atoms_per_molecule: int, max_bond_length: float, output_folder: str,
         contact_distance: float, merge_to_dimers: bool, drop_duplicates: bool):
    if not os.path.isdir(output_folder):
        raise OSError(f'{output_folder} is not a folder. Please provide a correct folder path')
    molecules_delimiter(input_file, min_atoms_per_molecule, max_bond_length, output_folder, contact_distance,
                        merge_to_dimers, drop_duplicates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='Input .xyz cartesian file')
    parser.add_argument('--min_atoms_per_molecule', type=int)
    parser.add_argument('--max_bond_length', type=float)
    parser.add_argument('--merge_to_dimers', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--contact_distance', type=float, default=4)
    parser.add_argument('--drop_duplicates', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output_folder')
    args = parser.parse_args()
    main(**vars(args))
