import argparse
import os
from glob import glob


def method_changer(input_file: str, method_file: str, output_file: str):
    """
    Change header in the input_file (everything except $DATA and $VEC groups)
    :param input_file: input file with .inp extension
    :param method_file: file with replacement header
    :param output_file: output .inp file
    :return:
    """

    # Read new header
    output_lines = []
    with open(method_file, encoding='utf8') as f:
        output_lines.extend(line for line in f)

    if not output_lines[-1].endswith('\n'):
        output_lines[-1] += '\n'

    # Read read input file after header ($DATA and $VEC groups)
    data_found = False
    with open(input_file, encoding='utf8') as f:
        for line in f:
            if not data_found:
                if '$DATA' in line.upper():
                    data_found = True
                    output_lines.append(line)
            else:
                output_lines.append(line)

    with open(output_file, 'w', encoding='utf8') as f:
        f.writelines(output_lines)


def main(input_path: str, method_file: str, output_path: str):
    if os.path.isdir(input_path):
        for input_file in glob(os.path.join(input_path, '*.inp')):
            output_file = os.path.join(output_path, os.path.basename(input_file))
            method_changer(input_file, method_file, output_file)
    elif (ext := os.path.splitext(input_path)[1]) == '.inp':
        output_file = os.path.join(output_path, os.path.basename(input_path))
        method_changer(input_path, method_file, output_file)
    else:
        raise TypeError(f'Unknown file format {ext}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Path to the folder with .inp files or single .inp file to process')
    parser.add_argument('--method_file', help='Path to the method file (header)')
    parser.add_argument('--output_path', help='Path to the output folder')
    args = parser.parse_args()
    main(**vars(args))
