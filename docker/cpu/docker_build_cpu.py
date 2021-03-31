#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import shutil
import getpass


def execute_build(args):

    image = args.image + '-cpu'
    docker_file = 'no_nvidia.dockerfile'

    if not os.path.exists(docker_file):
        print('Dockerfile %s not found! Exiting' % docker_file)
        return

    user_name = getpass.getuser()
    image = user_name + '-' + image
    print('Saving to image name: ' + image)
    
    cmd = 'docker build --build-arg USER_NAME=%(user_name)s \
            --build-arg USER_PASSWORD=%(password)s \
            --build-arg USER_ID=%(user_id)s \
            --build-arg USER_GID=%(group_id)s' \
            %{'user_name': user_name, 'password': args.password, 'user_id': args.user_id, 'group_id': args.group_id}

    cmd += ' --network=host '
    if args.no_cache:
        cmd += '--no-cache '
    if args.as_root:
        cmd = 'sudo ' + cmd
    cmd += '-t %s -f %s .' % (image, docker_file)

    print('command = \n\n', cmd)

    if not args.dry_run:
        os.system(cmd)


if __name__ == '__main__':

    default_image_name = "mpalm-dev"

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image', type=str,
                        default=default_image_name,
                        help='name for new docker image')

    parser.add_argument('--no_cache', action='store_true',
                        help='True if should build without using cache')

    parser.add_argument('-d', '--dry_run', action='store_true',
                        help='True if we should only print the build command '
                             'without executing')

    parser.add_argument("-pw", "--password", type=str,
                        help="(optional) password for the user", default="password")

    parser.add_argument('-uid','--user_id', type=int, help="(optional) user id for this user", default=os.getuid())
    parser.add_argument('-gid','--group_id', type=int, help="(optional) user gid for this user", default=os.getgid())
    parser.add_argument('--as_root', action='store_true')

    args = parser.parse_args()
    execute_build(args)

