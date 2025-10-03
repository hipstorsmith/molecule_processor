import abc
import argparse
import os
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree, distance_matrix
from scipy.optimize import linear_sum_assignment
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
        distance_fingerprint:
            sorted list of pairwise distances between atoms for quick comparison
        centered_coords:
            list of translated to (0, 0, 0) centroid atom coordinates
        atom_indices:
            dict of atom indices corresponding to each atom
    """

    def __init__(self, monomer_1: Monomer, monomer_2: Monomer):
        super().__init__()
        self.atoms = sorted(monomer_1.atoms + monomer_2.atoms, key=lambda a: a.idx)
        self.atom_names = np.array([a.name for a in self.atoms], dtype=object)
        self.type_counts = Counter(atom.name for atom in self.atoms)
        self.distance_signature = np.sort(distance_matrix(self.centered_coords, self.centered_coords), axis=1)
        self.distance_fingerprint = np.sort(self.distance_signature.flatten())
        self.monomer_labels = np.array([0 if a in monomer_1.atoms else 1 for a in self.atoms], dtype=np.int8)

    @property
    def centered_coords(self) -> np.ndarray:
        """
        List of translated to (0, 0, 0) centroid atom coordinates
        """

        coord = np.vstack([atom.coord for atom in self.atoms])
        centroid = np.mean(coord, axis=0)
        return coord - centroid

    @property
    def atom_indices(self):
        """
        Dict of atom indices corresponding to each atom
        """

        groups: dict[str, list[int]] = {}
        for i, atom in enumerate(self.atoms):
            groups.setdefault(atom.name, []).append(i)
        return {k: np.array(v, dtype=int) for k, v in groups.items()}


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

    return [Dimer(monomers[i], monomers[j]) for (i, j) in sorted(monomer_pair_idxs)]


def kabsch(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """
    Kabsch algorithm. Compute the optimal rotation (without scaling, translation and reflection) matrix that aligns two
    centered matrices mat_a[n, 3] to mat_b[n, 3] using least-squares method.
    :param mat_a:
    :param mat_b:
    :return: reflection matrix
    """

    # Cross-covariance matrix
    covariance = mat_a.T @ mat_b

    # SVD
    u, s, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T

    # Enforce a rotation without reflection
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T

    return rotation


def pairwise_sq_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate L2-squared norm: (a - b)**2
    :param a:
    :param b:
    :return:

    """
    a_sq = np.sum(a * a, axis=1, keepdims=True)
    b_sq = np.sum(b * b, axis=1, keepdims=True).T
    a_b = a @ b.T

    # Max to zero to avoid float errors
    return np.maximum(a_sq + b_sq - 2.0 * a_b, 0.0)


def rmsd_with_perm(mat_a: np.ndarray, mat_b: np.ndarray, perm: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Given centered coordinates mat_a[n, 3] and mat_b[n, 3] and a mapping permutation matrix (A->B), reorder mat_a to
    the order of mat_b and compute optimal rotation (via Kabsch algorithm) and RMSD (root-mean-square distance)
    :param mat_a:
    :param mat_b:
    :param perm: permutation for mat_a indices
    :return: (rmsd, rotation)
    """

    mat_a_ordered = np.empty_like(mat_a)
    mat_a_ordered[perm] = mat_a  # place atom i of mat_b into row perm[i] to match mat_q's row ordering

    # calculate rotation
    rotation = kabsch(mat_a_ordered, mat_b)

    # rotate
    mat_a_rotated = mat_a_ordered @ rotation

    # calculate rmsd
    diff = mat_a_rotated - mat_b
    rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    return rmsd, rotation


def build_assignment_from_signatures(distance_signature_a: np.ndarray, distance_signature_b: np.ndarray,
                                     atom_indices_a: dict[str, np.ndarray], atom_indices_b: dict[str, np.ndarray]
                                     ) -> np.ndarray:
    """
    Build an A->B index mapping per atom name using Hungarian assignment on distance signatures.
    :param distance_signature_a:
        first matrix of pairwise distances between each pair of atoms
    :param distance_signature_b:
        second matrix of pairwise distances between each pair of atoms
    :param atom_indices_a:
        first dict of atom indices corresponding to each atom
    :param atom_indices_b:
        second dict of atom indices corresponding to each atom
    :return:
        array of length distance_signature_a.shape[0] where perm[i] is the matched index in B for atom i in A.
    """

    assert set(atom_indices_a) == set(atom_indices_b)

    perm = np.empty(distance_signature_a.shape[0], dtype=int)

    for name in atom_indices_a.keys():
        idx_a = atom_indices_a[name]  # indices in A for this atom type
        idx_b = atom_indices_b[name]  # indices in B for this atom type
        signature_name_a = distance_signature_a[idx_a]
        signature_name_b = distance_signature_b[idx_b]

        # optimization cost is euclidian (L2) distance between signature rows
        cost_sq = pairwise_sq_distance(signature_name_a, signature_name_b)
        row_ind, col_ind = linear_sum_assignment(cost_sq)
        # permute
        perm[idx_a[row_ind]] = idx_b[col_ind]

    return perm


def refine_rotate_assignment(centered_coords_a: np.ndarray, centered_coords_b: np.ndarray,
                             atom_indices_a: dict[str, np.ndarray], atom_indices_b: dict[str, np.ndarray],
                             rotation: np.ndarray) -> np.ndarray:
    """
    Rotate matrix A and rebuild an A->B index mapping per atom name using Hungarian assignment on centered coordinates.
    :param centered_coords_a:
        first list of centered atom coordinates
    :param centered_coords_b:
        second list of centered atom coordinates
    :param atom_indices_a:
        first dict of atom indices corresponding to each atom
    :param atom_indices_b:
        second dict of atom indices corresponding to each atom
    :param rotation:
        rotation matrix for matrix a
    :return:
        array of length centered_coords_a.shape[0] where perm[i] is the matched index in B for atom i in A.
    """

    # rotate matrix A
    centered_coords_a_rot = centered_coords_a @ rotation
    perm = np.empty(centered_coords_a.shape[0], dtype=int)

    for name in atom_indices_a.keys():
        idx_a = atom_indices_a[name]  # indices in A for this atom type
        idx_b = atom_indices_b[name]  # indices in B for this atom type
        coord_name_a = centered_coords_a_rot[idx_a]
        coord_name_b = centered_coords_b[idx_b]

        # optimization cost is euclidian (L2) distance between two sets of coordinates
        cost_sq = pairwise_sq_distance(coord_name_a, coord_name_b)
        row_ind, col_ind = linear_sum_assignment(cost_sq)

        # permute
        perm[idx_a[row_ind]] = idx_b[col_ind]

    return perm


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

    for i, candidate in enumerate(dimers):
        is_duplicate = False
        if candidate.centered_coords.shape[0] <= 2:
            # Drop this dimer, there is clearly something wrong with it
            continue
        for j, dimer in enumerate(unique_dimers):
            # Checking amounts of atoms of different types. Not matching - not a duplicate
            if candidate.type_counts != dimer.type_counts:
                continue
            # If amount of atoms are matching - total amount of atoms is equal. Pairwise distances can be compared
            # Checking if sorted lists of pairwise distances are close. If they are not - not a duplicate
            if not np.allclose(candidate.distance_fingerprint, dimer.distance_fingerprint, atol=eps):
                continue

            # If distance fingerprints are matching - create an assignment using Hungarian algorithm matching
            init_permutation = build_assignment_from_signatures(candidate.distance_signature, dimer.distance_signature,
                                                                candidate.atom_indices, dimer.atom_indices)
            # Calculate current distance and rotation between candidate and current unique dimer
            init_rmsd, init_rotation = rmsd_with_perm(candidate.centered_coords, dimer.centered_coords,
                                                      init_permutation)
            if init_rmsd < eps:
                is_duplicate = True
                break

            # Rotate and recalculate permutation
            refined_permutation = refine_rotate_assignment(candidate.centered_coords, dimer.centered_coords,
                                                           candidate.atom_indices, dimer.atom_indices, init_rotation)
            # Calculate distance with new permutation
            refined_rmsd, _ = rmsd_with_perm(candidate.centered_coords, dimer.centered_coords, refined_permutation)
            if refined_rmsd < eps:
                is_duplicate = True
                break

            pass

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
