import argparse
import mendeleev
import numpy as np
import re
from collections import deque
from typing import List, Tuple
from itertools import groupby, compress
from operator import itemgetter

RADIUS_TABLE = {atom.symbol: (atom.covalent_radius or 246) / 100 for atom in mendeleev.get_all_elements()}
HEAD = """ $CONTRL RUNTYP=energy SCFTYP=RHF dfttyp=none
  MAXIT=200 ICHARG=0 MULT=1 d5=.t. nzvar=270
  exetyp=check coord=zmt
 $END
 $SYSTEM TIMLIM=3600000 MWORDS=350 $END
 $smp smppar=.t. load=0 call64=.t. $end
 $p2p p2p=.t. dlb=.t. mixed=.t. $end
 $trans mptran=2 dirtrf=.t. aoints=dist altpar=.t. mode=112 $end 
 $BASIS  GBASIS=n31 ngauss=6 NDFUNC=1 NPFUNC=1 $END
 $SCF DIRSCF=.TRUE. SOSCF=.f. $END
! $GUESS GUESS=moread $END  
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
        init_proc_data_var:
            list of tuples, containing calculated distance (for all atoms, except first), angle (for all atoms, except
            first two) and dihedral (for all atoms, except first three) for initial atom coordinates
        trans_proc_data_var:
            list of tuples, containing calculated distance (for all atoms, except first), angle (for all atoms, except
            first two) and dihedral (for all atoms, except first three) for transformed atom coordinates
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
            self.init_proc_data_var, self.trans_proc_data_var)
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
                should be shorter than 4 nodes (including current) and should not include atoms with one connection
                only, because of their instability
                args:
                    graph:
                return:
                    list of atoms, connected to chain
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
        self.init_proc_data_var = []
        self.trans_proc_data_var = []

    def connect(self, other: 'Atom'):
        """
        Connect Atom with the other by adding self.init_idx to other atom connections and vice versa
        :param other:
        """

        if not isinstance(other, Atom):
            raise TypeError(f'Expected type Atom, got {type(other)}')
        self.connections.add(other.init_idx)
        other.connections.add(self.init_idx)

    def calculate_data(self, queue_atoms):
        """
        Get all possible measurements for Atom, based on queue_atoms (fill self.proc_data_idx, self.init_proc_data_var,
        self.trans_proc_data_var)
        :param queue_atoms: atoms, based on which all connections should be performed
        """

        if len(queue_atoms) >= 1:
            self.proc_data_idx.append(queue_atoms[0].final_idx)
            self.init_proc_data_var.append((
                f'R{self.final_idx}_{queue_atoms[0].final_idx}',
                self.distance(queue_atoms[0], 'init_coord')))
            self.trans_proc_data_var.append((
                f'R{self.final_idx}_{queue_atoms[0].final_idx}',
                self.distance(queue_atoms[0], 'trans_coord')))
        if len(queue_atoms) >= 2:
            self.proc_data_idx.append(queue_atoms[1].final_idx)
            self.init_proc_data_var.append((
                f'A{self.final_idx}_{queue_atoms[0].final_idx}_{queue_atoms[1].final_idx}',
                self.angle(queue_atoms[0], queue_atoms[1], 'init_coord')))
            self.trans_proc_data_var.append((
                f'A{self.final_idx}_{queue_atoms[0].final_idx}_{queue_atoms[1].final_idx}',
                self.angle(queue_atoms[0], queue_atoms[1], 'trans_coord')))
        if len(queue_atoms) == 3:
            self.proc_data_idx.append(queue_atoms[2].final_idx)
            self.init_proc_data_var.append((
                f'D{self.final_idx}_{queue_atoms[0].final_idx}_{queue_atoms[1].final_idx}_'
                f'{queue_atoms[2].final_idx}',
                self.dihedral(queue_atoms[0], queue_atoms[1], queue_atoms[2], 'init_coord')))
            self.trans_proc_data_var.append((
                f'D{self.final_idx}_{queue_atoms[0].final_idx}_{queue_atoms[1].final_idx}_'
                f'{queue_atoms[2].final_idx}',
                self.dihedral(queue_atoms[0], queue_atoms[1], queue_atoms[2], 'trans_coord')))

    def distance(self, other: 'Atom', attr_name: str) -> float:
        """
        Calculate distance between Atom and other Atom
        :param other:
        :param attr_name: which coordinates should be used: 'init_coord' or 'trans_coord'
        :return: distance
        """

        if attr_name not in ('init_coord', 'trans_coord'):
            raise AttributeError(f"Invalid attribute name: '{attr_name}'. Must be 'init_coord' or 'trans_coord'.")
        return np.linalg.norm(getattr(self, attr_name) - getattr(other, attr_name))

    def angle(self, other_1: 'Atom', other_2: 'Atom', attr_name: str) -> float:
        """
        Calculate angle: self-other_1-other_2
        :param other_1:
        :param other_2:
        :param attr_name: which coordinates should be used: 'init_coord' or 'trans_coord'
        :return: angle in degrees
        """

        if attr_name not in ('init_coord', 'trans_coord'):
            raise AttributeError(f"Invalid attribute name: '{attr_name}'. Must be 'init_coord' or 'trans_coord'.")

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

        if attr_name not in ('init_coord', 'trans_coord'):
            raise AttributeError(f"Invalid attribute name: '{attr_name}'. Must be 'init_coord' or 'trans_coord'.")

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

        return np.degrees(np.arctan2(y, x))

    def route(self, graph) -> List['Atom']:
        """
        Calculate the longest possible route in graph, required for calculation of output parameters. Route should be
        shorter than 4 nodes (including current) and should not include atoms with one connection only, because of
        their instability
        :param graph:
        :return: list of atoms, connected to chain
        """

        longest_path = []
        stack = [(self, [self])]

        # TODO: add extra checks for 180 angles and small graphs

        while stack:
            current_node, path = stack.pop()
            if len(path) > len(longest_path):
                longest_path = path.copy()
            for neighbor in filter(lambda atom: atom.init_idx in current_node.connections, graph):
                if neighbor not in path and neighbor.name not in VALENCY_ONE:
                    new_path = path + [neighbor]
                    if len(new_path) <= 4:
                        stack.append((neighbor, new_path))

        return longest_path[1:]


def process_atoms_list(atoms_list: List[Atom]) -> Tuple[List[Atom], List[int]]:
    """
    Calculate all output data for all atoms in atoms_list using DFS and maximum column widths for all columns in
    output file
    :param atoms_list:
    :return: atoms_list, column_widths
    """

    column_widths = [0, 0, 0]
    new_index = 1
    visited = [False] * len(atoms_list)
    processing_queue = deque()

    # Find first atom with valency > 1
    start_idx = 0
    while atoms_list[start_idx].name in VALENCY_ONE:
        start_idx += 1
    processing_queue.append(atoms_list[start_idx])

    while processing_queue:
        atom = processing_queue.pop()
        atom.final_idx = new_index
        visited[atom.init_idx] = True

        # Find all atoms, based on which all the data for current atom will be calculated
        route = atom.route(list(compress(atoms_list, visited)))
        atom.calculate_data(route)

        # Update string formats
        column_widths = [a if a > b else b for a, b in
                       zip(column_widths, [len(col[0]) for col in atom.init_proc_data_var] + [0] * (
                               len(column_widths) - len(atom.init_proc_data_var)))]

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

    def dfs(node_index: int, component: List[int]):
        stack = [node_index]
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                component.append(current)
                for neighbor in atoms_list[current].connections:
                    if not visited[neighbor]:
                        stack.append(neighbor)

    for i in range(len(atoms_list)):
        if not visited[i]:
            component = []
            dfs(i, component)
            subgraphs.append(component)

    return subgraphs


def find_closest_pair_between_subgraphs(atoms_list: List[Atom], subgraphs: List[List[int]]) -> List[dict]:
    """
    Get a pair of atoms from different subgraphs, which have shortest distance
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


def main(xyz_file_start, xyz_file_end, zmt_file_start_out, zmt_file_end_out, debug_connections_out, coef):
    # read original xyz files without header
    with open(xyz_file_start, encoding='utf8') as f:
        xyz_start_coord = f.readlines()
        title = xyz_start_coord[1].strip()
        xyz_start_coord = xyz_start_coord[2:]
    with open(xyz_file_end, encoding='utf8') as f:
        xyz_end_coord = f.readlines()[2:]

    # saving initial and transformed coordinates into data structure
    atoms_list = [Atom(i, start_line, end_line) for i, (start_line, end_line) in
                  enumerate(zip(xyz_start_coord, xyz_end_coord))]

    # create graph connections (except for hydrogens)
    for atom1 in atoms_list:
        for atom2 in atoms_list:
            if (atom1.name not in VALENCY_ONE and atom2.name not in VALENCY_ONE and
                    atom1.init_idx != atom2.init_idx and
                    atom1.distance(atom2, 'init_coord') <= coef * (
                            RADIUS_TABLE[atom1.name] + RADIUS_TABLE[atom2.name])):
                atom1.connect(atom2)

    while len(subgraphs := find_subgraphs(atoms_list)) != 1:
        min_distances = find_closest_pair_between_subgraphs(atoms_list, subgraphs)
        for distance in min_distances:
            print(f"Graph {distance['graph1']}, atom {atoms_list[distance['index1']].name} "
                  f"({list(atoms_list[distance['index1']].init_coord)}) connected "
                  f"with graph {distance['graph2']}, atom {atoms_list[distance['index2']].name} "
                  f"({list(atoms_list[distance['index2']].init_coord)})")
            atoms_list[distance['index1']].connect(atoms_list[distance['index2']])

    with open(debug_connections_out, 'w', encoding='utf8') as f:
        f.write('source_index,name,initial_coord,trans_coord,connections\n')
        f.writelines(
            f'{atom.init_idx},{atom.name},{atom.init_coord},{atom.trans_coord},{sorted(atom.connections)}\n' for atom in
            atoms_list)

    atoms_list, column_widths = process_atoms_list(atoms_list)
    int_format = len(str(len(atoms_list)))
    with (open(zmt_file_start_out, 'w', encoding='utf8') as f_start,
          open(zmt_file_end_out, 'w', encoding='utf8') as f_end):
        f_start.write(HEAD.format(title=title))
        f_end.write(HEAD.format(title=title))
        for atom in atoms_list:
            start_line = '  '.join(
                f"{str(idx).ljust(int_format)}  {str(var[0]).ljust(column_widths[i])}" for i, (idx, var) in
                enumerate(zip(atom.proc_data_idx, atom.init_proc_data_var)))
            start_line = f'{atom.name}  {start_line}\n'
            f_start.write(start_line)

            end_line = '  '.join(
                f"{str(idx).ljust(int_format)}  {str(var[0]).ljust(column_widths[i])}" for i, (idx, var) in
                enumerate(zip(atom.proc_data_idx, atom.trans_proc_data_var)))
            end_line = f'{atom.name}  {end_line}\n'
            f_end.write(end_line)
        f_start.write('\n')
        f_end.write('\n')
        for atom in atoms_list:
            for var in atom.init_proc_data_var:
                f_start.write(f"{var[0].ljust(max(column_widths))}  =  {var[1]:.7f}\n")
            for var in atom.trans_proc_data_var:
                f_end.write(f"{var[0].ljust(max(column_widths))}  =  {var[1]:.7f}\n")

        f_start.write(START_MATRIX)
        f_end.write(START_MATRIX)

        for atom in atoms_list:
            for i, var in enumerate(atom.init_proc_data_var):
                f_start.write(" " * 12 + f"{i + 1}," + "".join(
                    f"  {idx.rjust(int_format)}," for idx in re.sub(r"[a-zA-Z]", "", var[0]).split("_")) + "\n")
            for i, var in enumerate(atom.trans_proc_data_var):
                f_end.write(" " * 12 + f"{i + 1}," + "".join(
                    f"  {idx.rjust(int_format)}," for idx in re.sub(r"[a-zA-Z]", "", var[0]).split("_")) + "\n")
        f_start.write(" $END \n")
        f_end.write(" $END \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xyz-file-start')
    parser.add_argument('--xyz-file-end')
    parser.add_argument('--zmt-file-start-out')
    parser.add_argument('--zmt-file-end-out')
    parser.add_argument('--debug-connections-out', default=None)
    parser.add_argument('--coef', type=float, default=1)

    args = parser.parse_args()
    main(**vars(args))
