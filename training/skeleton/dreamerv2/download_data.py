import os, os.path as osp
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_save_dir', type=str, required=True)
    parser.add_argument('--data_dir_name', type=str, required=True)
    parser.add_argument('--url', type=str, required=True)
    parser.add_argument('--dry_run', '-d', action='store_true')

    args = parser.parse_args()

    data_dir = osp.join(os.environ['CODE_BASE'], args.data_save_dir)
    if not osp.exists(data_dir):
        os.makedirs(data_dir)

    data_path = osp.join(data_dir, args.data_dir_name)
    cmd = f"wget -O {data_path} {args.url}; cd {data_dir}; tar -xvf {args.data_dir_name}"

    if not args.dry_run:
        os.system(cmd)
    else:
        print(cmd)
