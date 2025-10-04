import argparse
import mendeleev
import numpy as np
import re
import os
from collections import deque
from typing import List, Tuple, AnyStr
from itertools import groupby, compress, permutations
from operator import itemgetter

RADIUS_TABLE = {atom.symbol: (atom.covalent_radius or 246) / 100 for atom in mendeleev.get_all_elements()}
HEAD = """$CONTRL RUNTYP=energy SCFTYP=MCSCF dfttyp=none
  MAXIT=200 ICHARG={1,-1} MULT=2 d5=.t. nzvar={nzvar}
  exetyp={check,run} coord={coord}
 $END
 $SYSTEM TIMLIM=3600000 MWORDS=350 $END
 $smp smppar=.t. load=0 call64=.t. $end
 $p2p p2p=.t. dlb=.t. mixed=.t. $end
 $trans mptran=2 dirtrf=.t. aoints=dist altpar=.t. mode=112 $end 
 $BASIS  GBASIS=n31 ngauss=6 NDFUNC=1 NPFUNC=1 $END
 $SCF DIRSCF=.TRUE. SOSCF=.f. $END
{!} $GUESS GUESS=moread norb={norb} $END
 $DET NCORE={ncore} NACT=2 NELS={1,3} nstate=2 iroot=1 ITERMX=500 PURES=.t.             
  wstate(1)=1.,1.                                                               
  distci=64                                                                     
 $END                                                                           
 $ciinp castrf=.t. $end                                                         
 $MCSCF method=dm2 cistep=aldet                                                 
  SOSCF=.t.                                                                     
  istate=1                                                                      
  CASHFT=1.0 CASDII=0.01 MICIT=10                                               
  maxit=40                                                                      
 $END                                                                           
 $rimp2 auxbas=def2-TZVP/C extfil=.t. rimode=2 $end                             
 $XMCQDPT INORB=1 edshft=0.02 alttrf=.t. alttrf(1)=1,1,1,1                      
  nstate=2 ri=.t.                                                               
  wstate(1)=1,1,-0 avecoe(1)=1,1,-0 $end                                        
 $mcqfit $end                                                                   
 $DATA
{title}
C1
"""

START_MATRIX = """ $END      
 $ZMAT   IZMAT(1)=
"""

VALENCY_ONE = frozenset({'H', 'F', 'Cl', 'Br', 'I'})


class Atom:
    """
    Atom class
    attributes:
        init_idx:
            index in initial .xyz file
        final_idx:
            index in final .inp file
        name:
            atom name
        init_coord:
            initial atom coordinates
        trans_coord:
            transformed atom coordinates
        connections:
            set of init_idx of atoms, connected to current atom
        proc_data_idx:
            list of final_idx of atoms, on which distance, angle and dihedral are calculated
        proc_data_var:
            list of variable (distance, angle, dihedral) names in output file
        init_proc_data_value:
            list of calculated distances (for all atoms, except first), angles (for all atoms, except first two) and
            dihedrals (for all atoms, except first three) for initial atom coordinates
        trans_proc_data_value:
            list of calculated distances (for all atoms, except first), angles (for all atoms, except first two) and
            dihedrals (for all atoms, except first three) for transformed atom coordinates
        interpolation_points:
            interpolated values of distance, angle and dihedral between initial and transformed file
    methods:
        __init__(self, idx, initial_atom_line, trans_atom_line):
            args:
                idx:
                    atom index in original file
                initial_atom_line:
                    atom info line in original file
                trans_atom_line:
                    atom info line in transformed file
        connect(self, other):
            Connect atom with the other by adding self.init_idx to other.connections and vice versa
            args:
                other (Atom)
        calculate_data:
            Get all possible measurements for Atom, based on queue_atoms (fill self.proc_data_idx,
            self.init_proc_data_value, self.trans_proc_data_value)
            args:
                queue_atoms:
                    atoms, based on which all connections should be performed
        distance:
            Calculate distance between Atom and other Atom
            args:
                other:
                attr_name:
                    which coordinates should be used: 'init_coord' or 'trans_coord'
            return:
                distance
        angle:
            Calculate angle: self-other_1-other_2
            args:
                other_1:
                other_2:
                radians: calculate angle in radians
                attr_name: which coordinates should be used: 'init_coord' or 'trans_coord'
            return:
                angle in degrees
        dihedral:
            Calculate dihedral: self-other_1-other_2-other_3
            args:
                other_1:
                other_2:
                other_3:
                attr_name: which coordinates should be used: 'init_coord' or 'trans_coord'
            return:
                dihedral in degrees
            route
                Calculate the longest possible route in graph, required for calculation of output parameters. Route
                should be shorter than 4 atoms (including current) and should not include atoms with one connection
                only, because of their instability
                args:
                    graph:
                return:
                    list of atoms, connected to chain
        interpolate(self, n_points):
            Calculate interpolation points between initial and transformed atom distance, angle and dihedral
            args:
                n_points: number of interpolation points
    """

    def __init__(self, idx: int, initial_atom_line: str, trans_atom_line: str):
        """
        :param idx:
            atom index in original file
        :param initial_atom_line:
            atom info line in original file
        :param trans_atom_line:
            atom info line in transformed file
        """

        self.init_idx = idx
        self.final_idx = None
        self.name, self.init_coord = initial_atom_line.split(maxsplit=1)
        _, self.trans_coord = trans_atom_line.split(maxsplit=1)
        self.init_coord = np.array([float(x) for x in self.init_coord.split()])
        self.trans_coord = np.array([float(x) for x in self.trans_coord.split()])
        self.connections = set()
        self.proc_data_idx = []
        self.proc_data_var = []
        self.init_proc_data_value = []
        self.trans_proc_data_value = []
        self.interpolation_points = []

    def connect(self, other: 'Atom'):
        """
        Connect Atom with the other by adding self.init_idx to other atom connections and vice versa
        :param other:
        """

        assert isinstance(other, Atom), f'Expected type Atom, got {type(other)}'
        self.connections.add(other.init_idx)
        other.connections.add(self.init_idx)

    def calculate_data(self, queue_atoms, n_points):
        """
        Get all possible measurements for Atom, based on queue_atoms (fill self.proc_data_idx, self.init_proc_data_value,
        self.trans_proc_data_value)
        :param queue_atoms: atoms, based on which all connections should be performed
        :param n_points: number of interpolation points
        """

        if len(queue_atoms) >= 1:
            self.proc_data_idx.append(queue_atoms[0].final_idx)
            self.proc_data_var.append(f'R{self.final_idx}_{queue_atoms[0].final_idx}')
            self.init_proc_data_value.append(self.distance(queue_atoms[0], 'init_coord'))
            self.trans_proc_data_value.append(self.distance(queue_atoms[0], 'trans_coord'))
        if len(queue_atoms) >= 2:
            self.proc_data_idx.append(queue_atoms[1].final_idx)
            self.proc_data_var.append(f'A{self.final_idx}_{queue_atoms[0].final_idx}_{queue_atoms[1].final_idx}')
            self.init_proc_data_value.append(self.angle(queue_atoms[0], queue_atoms[1], 'init_coord'))
            self.trans_proc_data_value.append(self.angle(queue_atoms[0], queue_atoms[1], 'trans_coord'))
        if len(queue_atoms) == 3:
            self.proc_data_idx.append(queue_atoms[2].final_idx)
            self.proc_data_var.append(f'D{self.final_idx}_{queue_atoms[0].final_idx}_{queue_atoms[1].final_idx}_'
                                      f'{queue_atoms[2].final_idx}')
            self.init_proc_data_value.append(
                self.dihedral(queue_atoms[0], queue_atoms[1], queue_atoms[2], 'init_coord'))
            self.trans_proc_data_value.append(
                self.dihedral(queue_atoms[0], queue_atoms[1], queue_atoms[2], 'trans_coord'))
        self.interpolate(n_points)
        self.init_proc_data_value = [self.init_proc_data_value]
        self.trans_proc_data_value = [self.trans_proc_data_value]

    def distance(self, other: 'Atom', attr_name: str) -> float:
        """
        Calculate distance between Atom and other Atom
        :param other:
        :param attr_name: which coordinates should be used: 'init_coord' or 'trans_coord'
        :return: distance
        """

        assert attr_name in (
            'init_coord',
            'trans_coord'), f"Invalid attribute name: '{attr_name}'. Must be 'init_coord' or 'trans_coord'"
        return np.linalg.norm(getattr(self, attr_name) - getattr(other, attr_name))

    def angle(self, other_1: 'Atom', other_2: 'Atom', attr_name: str) -> float:
        """
        Calculate angle: self-other_1-other_2
        :param other_1:
        :param other_2:
        :param attr_name: which coordinates should be used: 'init_coord' or 'trans_coord'
        :return: angle in degrees
        """

        assert attr_name in (
            'init_coord',
            'trans_coord'), f"Invalid attribute name: '{attr_name}'. Must be 'init_coord' or 'trans_coord'"

        # Calculate vectors self-other_1 and other_2-other_1
        vec1 = getattr(self, attr_name) - getattr(other_1, attr_name)
        vec2 = getattr(other_2, attr_name) - getattr(other_1, attr_name)

        return np.degrees(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))

    def dihedral(self, other_1: 'Atom', other_2: 'Atom', other_3: 'Atom', attr_name: str) -> float:
        """
        Calculate dihedral: self-other_1-other_2-other_3
        :param other_1:
        :param other_2:
        :param other_3:
        :param attr_name: which coordinates should be used: 'init_coord' or 'trans_coord'
        :return: dihedral in degrees
        """

        assert attr_name in (
            'init_coord',
            'trans_coord'), f"Invalid attribute name: '{attr_name}'. Must be 'init_coord' or 'trans_coord'"

        # Calculate vectors self-other_1, other_2-other_1 and other_3-other_2
        vec_1 = -np.subtract(getattr(other_1, attr_name), getattr(self, attr_name))
        vec_2 = np.subtract(getattr(other_2, attr_name), getattr(other_1, attr_name))
        vec_3 = np.subtract(getattr(other_3, attr_name), getattr(other_2, attr_name))

        # normalize vec_2
        vec_2 /= np.linalg.norm(vec_2)

        # calculate projections onto planes
        v = vec_1 - np.dot(vec_1, vec_2) * vec_2
        w = vec_3 - np.dot(vec_3, vec_2) * vec_2

        # calculate angle between projections
        x = np.dot(v, w)
        y = np.dot(np.cross(vec_2, v), w)

        return np.degrees(np.arctan2(y, x))  # self.fix_dihedral(np.degrees(np.arctan2(y, x)))

    @staticmethod
    def validate_angle(path: List['Atom']):
        if len(path) >= 3:
            atom_a, atom_b, atom_c = path[-3], path[-2], path[-1]
            return atom_a.angle(atom_b, atom_c, 'init_coord') not in (0, 180)
        return True

    def validate_atom(self, atom, path, allow_valency_one):
        """Checks if the atom is valid to be added to the path based on FORBIDDEN and angle conditions."""
        if atom in path or atom == self:
            return False
        if not allow_valency_one and atom.name in VALENCY_ONE:
            return False
        return True

    def fallback_atoms(self, fallback_atoms, max_len):
        for atom_selection in sorted(permutations(fallback_atoms, max_len - 1),
                                     key=lambda selection: sum(atom.name not in VALENCY_ONE for atom in selection),
                                     reverse=True):
            if len(atom_selection) < 2:
                return [self] + [atom_selection]
            else:
                if self.validate_angle([self] + list(atom_selection)[:-1]) and self.validate_angle(
                        [self] + list(atom_selection)):
                    return [self] + [atom_selection]

        return []

    def route(self, graph: List['Atom'], allow_valency_one: bool = False) -> List['Atom']:
        """
        Calculate the longest possible route in graph, required for calculation of output parameters. Route should be
        shorter than 4 atoms (including current) and should not include atoms with one connection only, because of
        their instability
        :param graph: list of already processed atoms and atoms already in
        :param allow_valency_one: allow anchoring on atoms with valency = 1
        :return: list of atoms, connected to chain
        """

        required_length = 4 if self.final_idx > 3 else self.final_idx

        longest_path = []
        stack = [(self, [self])]

        while stack:
            current_atom, path = stack.pop()
            if len(path) == required_length and self.validate_angle(path):
                if len(path) > len(longest_path):
                    longest_path = path.copy()
            for neighbor in filter(lambda atom: atom.init_idx in current_atom.connections, graph):
                if self.validate_atom(neighbor, path, allow_valency_one):
                    new_path = path + [neighbor]
                    if self.validate_angle(new_path) and len(new_path) <= required_length:
                        stack.append((neighbor, new_path))

        if len(longest_path) == required_length:
            return longest_path[1:]

        fallback_atoms = [atom for atom in graph if self.validate_atom(atom, [], allow_valency_one)]
        selected_path = self.fallback_atoms(fallback_atoms, required_length)
        if len(selected_path) == required_length:
            return selected_path[1:]  # Exclude initial atom

        fallback_atoms = [atom for atom in graph if atom not in selected_path and atom != self]
        selected_path = self.fallback_atoms(fallback_atoms, required_length)
        if len(selected_path) == required_length:
            return selected_path[1:]

        selected_path = [self] + sorted([atom for atom in graph if atom != self],
                                        key=lambda atom: atom.name in VALENCY_ONE)[:required_length - 1]
        return selected_path[1:]

    def interpolate(self, n_points):
        """
        Calculate interpolation points between initial and transformed atom distance, angle and dihedral
        :param n_points: number of interpolation points
        :return:
        """

        if (len(self.trans_proc_data_value) == 3) and (
                abs(self.trans_proc_data_value[2] - self.init_proc_data_value[2]) >= 180):
            self.init_proc_data_value[2] = self.init_proc_data_value[2] + 360 if self.init_proc_data_value[2] < 0 else \
            self.init_proc_data_value[2]
            self.trans_proc_data_value[2] = self.trans_proc_data_value[2] + 360 if self.trans_proc_data_value[
                                                                                       2] < 0 else \
            self.trans_proc_data_value[2]
        for var_idx, (start, end) in enumerate(zip(self.init_proc_data_value, self.trans_proc_data_value)):
            points = np.linspace(start, end, n_points + 2)[1:-1]
            self.interpolation_points.append(points.tolist())
        self.interpolation_points = list(map(list, zip(*self.interpolation_points)))
        self.interpolation_points.extend([[]] * (n_points - len(self.interpolation_points)))


def process_atoms_list(atoms_list: List[Atom], n_points: int) -> Tuple[List[Atom], List[int]]:
    """
    Calculate all output data for all atoms in atoms_list using DFS and maximum column widths for all columns in
    output file
    :param atoms_list:
    :param n_points: number of interpolation points
    :return: atoms_list, column_widths
    """

    column_widths = [0, 0, 0]
    new_index = 1
    visited = [False] * len(atoms_list)
    processing_queue = deque()

    allow_valency_one = sum(map(lambda a: a.name not in VALENCY_ONE, atoms_list)) <= 3

    # Find first atom with valency > 1
    start_idx = 0
    while atoms_list[start_idx].name in VALENCY_ONE:
        start_idx += 1
        if start_idx == len(atoms_list):
            break
    processing_queue.append(atoms_list[start_idx])

    while processing_queue:
        atom = processing_queue.pop()
        atom.final_idx = new_index
        visited[atom.init_idx] = True

        # Find all atoms, based on which all the data for current atom will be calculated
        route = atom.route(list(compress(atoms_list, visited)), allow_valency_one)
        atom.calculate_data(route, n_points)

        # Update string formats
        column_widths = [a if a > b else b for a, b in
                         zip(column_widths, [len(col) for col in atom.proc_data_var] + [0] * (
                                 len(column_widths) - len(atom.proc_data_var)))]

        # Add atom connections to queue
        for i in atom.connections:
            if not visited[i] and atoms_list[i] not in processing_queue:
                processing_queue.append(atoms_list[i])
        new_index += 1

    return sorted(atoms_list, key=lambda a: a.final_idx), column_widths


def find_subgraphs(atoms_list: List[Atom]) -> List[List[int]]:
    """
    Find all sub-graphs in graph
    :param atoms_list:
    :return:
    """
    visited = [False] * len(atoms_list)
    subgraphs = []

    for i in range(len(atoms_list)):
        if not visited[i]:
            component = []
            stack = [i]
            while stack:
                current = stack.pop()
                if not visited[current]:
                    visited[current] = True
                    component.append(current)
                    for neighbor in atoms_list[current].connections:
                        if not visited[neighbor]:
                            stack.append(neighbor)
            subgraphs.append(component)

    return subgraphs


def find_closest_pair_between_subgraphs(atoms_list: List[Atom], subgraphs: List[List[int]]) -> List[dict]:
    """
    Get a pair of atoms from different subgraphs, which have the shortest distance
    :param atoms_list:
    :param subgraphs:
    :return:
    """
    min_distances = []

    for i in range(len(subgraphs)):
        for j in range(len(subgraphs)):
            if i == j:
                continue
            subgraph1 = subgraphs[i]
            subgraph2 = subgraphs[j]
            min_distance = float('inf')
            min_index1 = float('inf')
            min_index2 = float('inf')

            for index1 in subgraph1:
                for index2 in subgraph2:
                    distance = atoms_list[index1].distance(atoms_list[index2], 'init_coord')
                    if distance < min_distance and atoms_list[index2].name not in VALENCY_ONE:
                        min_distance = distance
                        min_index1 = index1
                        min_index2 = index2
            if min_distance != float('inf'):
                min_distances.append({'graph1': i,
                                      'graph2': j,
                                      'index1': min_index1,
                                      'index2': min_index2,
                                      'distance': min_distance})

    min_distances = sorted(min_distances, key=lambda line: (line['graph1'], line['distance']))

    return [list(val)[0] for _, val in groupby(min_distances,
                                               key=itemgetter('graph1'))]


def write_z_matrix(file_path: AnyStr, title: str, atoms_list: List[Atom], int_format: int, column_widths: List[int],
                   coord_var_name: str, point_idx: int = 0):
    """
    Write z-matrix to file
    :param file_path: path to output file
    :param title: molecule title
    :param atoms_list: list of atoms
    :param int_format: length of integers in output file
    :param column_widths: widths of columns in output file
    :param coord_var_name: name of coordinate array in Atom to put into file
    :param point_idx: number of interpolation point in list (0 for initial and transformed)
    :return:
    """
    assert coord_var_name in ('init_proc_data_value', 'trans_proc_data_value', 'interpolation_points'), \
        (f"Invalid attribute name: '{coord_var_name}'. Must be 'init_proc_data_value', 'trans_proc_data_value' or"
         f" 'interpolation_points'")

    # TODO: coord='unique' when conversion to xyz

    with open(file_path, 'w', encoding='utf8') as f:
        f.write(HEAD.format(title=title, nzvar=3 * len(atoms_list) - 6, coord='zmt'))
        for atom in atoms_list:
            line = '  '.join(
                f"{str(idx).ljust(int_format)}  {str(var).ljust(column_widths[i])}" for i, (idx, var) in
                enumerate(zip(atom.proc_data_idx, atom.proc_data_var)))
            line = f'{atom.name}  {line}\n'
            f.write(line)

        f.write('\n')
        for atom in atoms_list:
            for var, value in zip(atom.proc_data_var, getattr(atom, coord_var_name)[point_idx]):
                f.write(f"{var.ljust(max(column_widths))}  =  {value:.7f}\n")

        f.write(START_MATRIX)

        for atom in atoms_list:
            for i, var in enumerate(atom.proc_data_var):
                f.write(" " * 12 + f"{i + 1}," + "".join(
                    f"  {idx.rjust(int_format)}," for idx in re.sub(r"[a-zA-Z]", "", var).split("_")) + "\n")
        f.write(" $END \n")

def run_interpolation(xyz_file_init, xyz_file_trans, zmt_folder_out, n_points):
    # read original xyz files without header
    with open(xyz_file_init, encoding='utf8') as f:
        xyz_init_coord = f.readlines()
        title = xyz_init_coord[1].strip()
        xyz_init_coord = xyz_init_coord[2:]
    with open(xyz_file_trans, encoding='utf8') as f:
        xyz_trans_coord = f.readlines()[2:]

    # saving initial and transformed coordinates into data structure
    atoms_list = [Atom(i, init_line, trans_line) for i, (init_line, trans_line) in
                  enumerate(zip(xyz_init_coord, xyz_trans_coord))]

    # create graph connections (except for hydrogens)
    for atom1 in atoms_list:
        for atom2 in atoms_list:
            if (atom1.name not in VALENCY_ONE and atom2.name not in VALENCY_ONE and
                    atom1.init_idx != atom2.init_idx and
                    atom1.distance(atom2, 'init_coord') <= RADIUS_TABLE[atom1.name] + RADIUS_TABLE[atom2.name]):
                atom1.connect(atom2)

    while len(subgraphs := find_subgraphs(atoms_list)) != 1:
        min_distances = find_closest_pair_between_subgraphs(atoms_list, subgraphs)
        for distance in min_distances:
            print(f"Graph {distance['graph1']}, atom {atoms_list[distance['index1']].name} "
                  f"({list(atoms_list[distance['index1']].init_coord)}) connected "
                  f"with graph {distance['graph2']}, atom {atoms_list[distance['index2']].name} "
                  f"({list(atoms_list[distance['index2']].init_coord)})")
            atoms_list[distance['index1']].connect(atoms_list[distance['index2']])

    atoms_list, column_widths = process_atoms_list(atoms_list, n_points)
    int_format = len(str(len(atoms_list)))

    write_z_matrix(file_path=os.path.join(zmt_folder_out, os.path.splitext(os.path.basename(xyz_file_init))[0] + '.000.inp'),
                   title=title,
                   atoms_list=atoms_list,
                   int_format=int_format,
                   column_widths=column_widths,
                   coord_var_name='init_proc_data_value')
    write_z_matrix(file_path=os.path.join(zmt_folder_out, os.path.splitext(os.path.basename(xyz_file_init))[0] + '.100.inp'),
                   title=title,
                   atoms_list=atoms_list,
                   int_format=int_format,
                   column_widths=column_widths,
                   coord_var_name='trans_proc_data_value')
    for i in range(1, n_points):
        coord = str(i * 100 // (n_points - 1)).zfill(3)
        filename = os.path.splitext(os.path.basename(xyz_file_init))[0] + f'.{coord}.inp'
        write_z_matrix(file_path=os.path.join(zmt_folder_out, filename),
                       title=title,
                       atoms_list=atoms_list,
                       int_format=int_format,
                       column_widths=column_widths,
                       coord_var_name='interpolation_points',
                       point_idx=i)

def main(xyz_file_init, xyz_file_trans, zmt_folder_out, n_points):
    run_interpolation(xyz_file_init, xyz_file_trans, zmt_folder_out, n_points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xyz-file-init', help='XYZ file with initial atom coordinates')
    parser.add_argument('--xyz-file-trans', help='XYZ file with transformed atom coordinates')
    parser.add_argument('--zmt-folder-out', help='Output folder for z-matrices')
    parser.add_argument('--n-points', type=int, default=0, help='Number of interpolation points')

    args = parser.parse_args()
    main(**vars(args))
