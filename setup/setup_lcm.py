import os, os.path as osp
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lcm-path', 
                        type=str, 
                        default=osp.join(os.environ['HOME'], 'lcm'))
    parser.add_argument('--virtualenv-path', '-venv',
                        type=str,
                        default=osp.join(os.environ['HOME'], 'environments/py36-gnn'))
    parser.add_argument('--dry_run', '-d', action='store_true')

    args = parser.parse_args()
    cmd = f"cd {args.lcm_path}/lcm-python; pwd; ls; python setup.py install"

    if not args.dry_run:
        os.system(cmd)
    else:
        print(cmd)