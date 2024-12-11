#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:24:29 2024

@author: jstout
"""

import os, os.path as op
import pandas as pd
from nih2mne.utilities import markerfile_write
from nih2mne.utilities.markerfile_write import main as write_markerfile

from nih2mne.utilities.trigger_utilities import (parse_marks, threshold_detect, 
    detect_digital)

import mne
import sys

fname = sys.argv[1]

#project_dir = '/fast2/SPIN'
#os.chdir(project_dir)
#fname = op.join(project_dir, '20241112', 'MEG_SPiN_20241112_001.ds')
#raw = mne.io.read_raw_ctf(fname, system_clock='ignore', preload=True)

dframe = detect_digital(fname)

auditory_delay = 0.048
dframe.onset += auditory_delay   #Add the auditory delay

condition_dict = {
                  '8':'...',    #<<<Fill these in with your condition naming
                  '16':'....',
                  '32':'',
                  '64':''
                  }

#Rename your events
for key,value in condition_dict.items():
    dframe.loc[dframe.condition==key, 'condition']=value

try:
    write_markerfile(dframe, fname)
    print(f'Wrote markers to {fname}')
except:
    print(f'Failed to write markerfile for {fname}')
