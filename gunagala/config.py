"""
The MIT License (MIT)

Copyright (c) 2014-2016 PANOPTES
Copyright 2016 Google Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import yaml

from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
from warnings import warn


def listify(obj):
    """ Given an object, return a list

    Always returns a list. If obj is None, returns empty list,
    if obj is list, just returns obj, otherwise returns list with
    obj as single member.

    Returns:
        list:   You guessed it.
    """
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, type(None))) else [obj]


def load_config(config_files=None, simulator=None, parse=True, ignore_local=False):
    """ Load configuation information """

    # Default to the pocs.yaml file
    if config_files is None:
        config_files = ['pocs']
    config_files = listify(config_files)

    config = dict()

    config_dir = 'data'

    for f in config_files:
        if not f.endswith('.yaml'):
            f = '{}.yaml'.format(f)

        if not f.startswith('/'):
            path = get_pkg_data_filename(os.path.join(config_dir, f))
        else:
            path = f

        try:
            _add_to_conf(config, path)
        except Exception as e:
            warn("Problem with config file {}, skipping. {}".format(path, e))

        # Load local version of config
        if not ignore_local:
            local_version = os.path.join(config_dir, f.replace('.', '_local.'))
            if os.path.exists(local_version):
                try:
                    _add_to_conf(config, local_version)
                except Exception:
                    warn("Problem with local config file {}, skipping".format(local_version))

    if simulator is not None:
        if 'all' in simulator:
            config['simulator'] = ['camera', 'mount', 'weather', 'night']
        else:
            config['simulator'] = simulator

    if parse:
        config = parse_config(config)

    return config


def parse_config(config):
    # Add units to our location
    if 'location' in config:
        loc = config['location']

        for angle in ['latitude', 'longitude', 'horizon', 'twilight_horizon']:
            if angle in loc:
                loc[angle] = loc[angle] * u.degree

        loc['elevation'] = loc.get('elevation', 0) * u.meter

    # Prepend the base directory to relative dirs
    if 'directories' in config:
        base_dir = os.getenv('PANDIR')
        for dir_name, rel_dir in config['directories'].items():
            if not rel_dir.startswith('/'):
                config['directories'][dir_name] = '{}/{}'.format(base_dir, rel_dir)

    return config


def save_config(path, config, clobber=True):
    if not path.endswith('.yaml'):
        path = '{}.yaml'.format(path)

    if not path.startswith('/'):
        config_dir = '{}/conf_files'.format(os.getenv('POCS'))
        path = os.path.join(config_dir, path)

    if os.path.exists(path) and not clobber:
        warn("Path exists and clobber=False: {}".format(path))
    else:
        with open(path, 'w') as f:
            f.write(yaml.dump(config))


def _add_to_conf(config, fn):
    try:
        with open(fn, 'r') as f:
            c = yaml.load(f.read())
            if c is not None and isinstance(c, dict):
                config.update(c)
    except IOError:  # pragma: no cover
        pass
