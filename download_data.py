"""Download data after cloning.

Run this once. It will create `data` and download the `train.pkl` and
`test.pkl` there.
"""

import os
import shutil
from subprocess import call

if os.path.exists('data'):
    shutil.rmtree('data')
os.mkdir('data')

url = 'https://storage.ramp.studio/radar_trajectories'
f_names = ['train.pkl', 'test.pkl']
for f_name in f_names:
    url_in = '{}/{}'.format(url, f_name)
    f_name_out = os.path.join('data', f_name)
    cmd = 'wget {} --output-document={} --no-check-certificate'.format(
        url_in, f_name_out)
    call(cmd, shell=True)
