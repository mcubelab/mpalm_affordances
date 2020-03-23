#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import shutil
import getpass


def execute_build(args):

    # copy requirements file from parent into docker folder
    cwd = os.getcwd()
    shutil.copy(cwd+'/../requirements.txt', cwd)

    if args.gpu:
        image = args.image + '-gpu'
        docker_file = 'nvidia.dockerfile'
    else:
        image = args.image + '-cpu'
        docker_file = 'no_nvidia.dockerfile'

    if args.pytorch:
        image = args.image + '-pytorch'
        docker_file = 'pytorch.dockerfile'

    if not os.path.exists(docker_file):
        print('Dockerfile %s not found! Exiting' % docker_file)
        return

    # cmd = 'docker build '
    user_name = getpass.getuser()
    cmd = 'docker build --build-arg USER_NAME=%(user_name)s \
            --build-arg USER_PASSWORD=%(password)s \
            --build-arg USER_ID=%(user_id)s \
            --build-arg USER_GID=%(group_id)s' \
            %{'user_name': user_name, 'password': args.password, 'user_id': args.user_id, 'group_id': args.group_id}

    cmd += ' --network=host '
    if args.no_cache:
        cmd += '--no-cache '
    cmd += '-t %s -f %s .' % (image, docker_file)

    print('command = \n\n', cmd)

    if not args.dry_run:
        os.system(cmd)

    # removing copied requirements file from docker/ directory
    os.remove('requirements.txt')


if __name__ == '__main__':

    default_image_name = "mpalm-dev"

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image', type=str,
                        default=default_image_name,
                        help='name for new docker image')

    parser.add_argument('--no_cache', action='store_true',
                        help='True if should build without using cache')

    parser.add_argument('-g', '--gpu', action='store_true',
                        help='True if we should build for use on'
                             'machine with nvidia gpu')

    parser.add_argument('-d', '--dry_run', action='store_true',
                        help='True if we should only print the build command '
                             'without executing')

    parser.add_argument('--pytorch', action='store_true')

    parser.add_argument("-pw", "--password", type=str,
                        help="(optional) password for the user", default="password")

    parser.add_argument('-uid','--user_id', type=int, help="(optional) user id for this user", default=os.getuid())
    parser.add_argument('-gid','--group_id', type=int, help="(optional) user gid for this user", default=os.getgid())

    args = parser.parse_args()
    execute_build(args)

