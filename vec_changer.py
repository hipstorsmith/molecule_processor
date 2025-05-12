import argparse
import os
from glob import glob


def vec_changer(input_file, vec_file, output_file):
    """
    Change $VEC group in the input_file
    :param input_file: input file with .inp extension
    :param vec_file: file with replacement $VEC group
    :param output_file: output .inp file
    :return:
    """

    # Read input file except $VEC group
    output_lines = []
    with open(input_file, encoding='utf8') as f:
        for line in f:
            if '$VEC' in line.upper():
                break
            output_lines.append(line)

    if not output_lines[-1].endswith('\n'):
        output_lines[-1] += '\n'

    # Read $VEC group
    vec_found = False
    with open(vec_file, encoding='utf8') as f:
        for line in f:
            if not vec_found:
                if '$VEC' in line.upper():
                    vec_found = True
                    output_lines.append(line)
            else:
                output_lines.append(line)

    with open(output_file, 'w', encoding='utf8') as f:
        f.writelines(output_lines)


def main(input_path, vec_file, output_path):
    if os.path.isdir(input_path):
        for input_file in glob(os.path.join(input_path, '*.inp')):
            output_file = os.path.join(output_path, os.path.basename(input_file))
            vec_changer(input_file, vec_file, output_file)
    elif (ext := os.path.splitext(input_path)[1]) == '.inp':
        output_file = os.path.join(output_path, os.path.basename(input_path))
        vec_changer(input_path, vec_file, output_file)
    else:
        raise TypeError(f'Unknown file format {ext}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Path to the folder with .inp files or single .inp file to process')
    parser.add_argument('--vec_file', help='Path to the vec file')
    parser.add_argument('--output_path', help='Path to the output folder')
    args = parser.parse_args()
    main(**vars(args))
