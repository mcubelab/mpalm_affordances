import trimesh
import os
import argparse


def main(args):
    data_dir = args.data_dir

    print('Loading from: ' + str(data_dir))
    raw_input('Press enter to continue\n')

    for fname in os.listdir(data_dir):
        if fname != 'nominal_cuboid.stl':
            tmesh = trimesh.load_mesh(os.path.join(data_dir, fname))
            tmesh.apply_scale(0.001)
            tmesh.export(os.path.join(data_dir, fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str
    )
    args = parser.parse_args()
    main(args)