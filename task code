from datetime import datetime
import numpy as np
import os
import pandas as pd
from psychopy import prefs, sound, visual, event, core, parallel, logging
from psychopy.hardware import keyboard
from psychopy.constants import (NOT_STARTED, STARTED)
import time
import glob
prefs.hardware['audioLib']  == 'ptb'
prefs.hardware['audioLatencyMode'] = 3

########## PARAMETERS ##########

participant = '01' # Participant details
part = 'A'
run = 0
ismeg = 0  # 1 = YES
isfullscreen = 1
iseyetracking = 0
is_dummy_test = 1

##
# Get experiment filepath
curr_path = os.path.abspath(os.path.dirname(__file__))
if is_dummy_test == 0:
    # Get sound path
    soundFile = os.path.join(curr_path, 'audio', f'pupil_part_{part}.wav')
    # Load csv file
    trigger_info = pd.read_csv(f'task_Triggers_part{part}.csv')
else:
    soundFile = os.path.join(curr_path, 'audio', f'Pupillometry_test.wav')

    # Create t_onset column with increments of 30s starting from 10s
    t_onset = [10 + i * 20 for i in range(5)]

    # Create t_offset column which is t_onset + 10
    t_offset = [t + 10 for t in t_onset]

    # Create DataFrame with condition names and time columns
    data = {
        'condition_name': ["A", "B", "C", "D", "E"],
        't_onset': t_onset,
        't_offset': t_offset,
        'status_onset': [0,0,0,0,0],
        'status_offset': [0,0,0,0,0]
    }

    trigger_info = pd.DataFrame(data)

# Define keys on the keyboard for control
response_keys = ['1', '2', '3', '4', 'q', 'Q']  # 'q', 'Q' for quitting program

# Current date and time for saving
now = datetime.now()
t = now.strftime("%m%d%Y_%H%M%S")

# Setting saving paths
results_path = os.path.join(curr_path, 'results')
if not os.path.exists(results_path):
    os.makedirs(results_path)

fname_data = os.path.join(results_path,f'S{participant.zfill(2)}P{part}_{t}')
if glob.glob(os.path.join(results_path,f'S{participant.zfill(2)}P{part}*')):
    raise ValueError(f"\n\nFilename starting with S{participant.zfill(2)}P{part} already exists\n\n")

# window display specs
background_col = [100, 100, 100] # rgb 255 space

######################################
########## HELPER FUNCTIONS ##########
######################################

# Eyetracker-related
if iseyetracking:
    print('eyetracking ON. loading necessary modules...')
    from eyetracking import *
    print('[done].')

# stimulus-related functions
def trigger(port,code):
    port.setData(int(code))

# trigger related
def setup_triggers():
    port = parallel.ParallelPort(address=0x0378)
    port.setData(0)
    return port

if ismeg:
    port = setup_triggers()
else:
    port = []

def draw_stim(win,stim,ismeg,trigger_code=100, port = []):
    stim.draw()
    if ismeg:
        win.callOnFlip(trigger,port = port,code=trigger_code)
    return win.flip()

def draw_text(text,win):
    text.draw()
    win.flip()

def instruction_continue(keyList):
    # Wait for key press
    key = event.waitKeys(keyList=keyList)

    # End experiment
    if key and key[0] in ['escape', 'q', 'Q']:
        exit_experiment()

def saveExpLog(snd,fname_data,quitPressed=0):

    # Assuming you have a DataFrame initialized (df) or you can create one
    # Here's a simple example with an empty DataFrame
    df = pd.DataFrame(columns=['fileName', 'volume', 'sampleRate', 'duration', 'frameNStart', 'tStart', 'tStartRefresh', 'tStop', 'frameNStop'])

    # Assuming 'snd' is an object with attributes you want to save
    for key, val in snd.__dict__.items():
        if key in df.columns:
            df[key] = [val]

    # Save the DataFrame to a CSV file
    if quitPressed:
        fileName = f'{fname_data}_quit.csv'
    else:
        fileName = f'{fname_data}.csv'
    df.to_csv(fileName, index=False)
    
def exit_experiment():

    print('User quit experiment')
    try:
        snd.stop() 
        saveExpLog(snd, fname_data, quitPressed=1)
        if iseyetracking:
            el_tracker.sendMessage(f'Experiment quit')
            exit(el_tracker, et_fname, results_folder=results_path)

    except Exception as e:
        print(f"Error during graceful exit: {e}")

    finally:
        win.close()
        core.quit()

##################################
##### SET UP WINDOW, STIMULI #####
##################################

if iseyetracking: # if on, do eyetracker setup 
    fname = f'{participant}P{str(part)}'
    et_fname = make_filename(fname=fname)
    el_tracker, win, genv, _ = connect(fname=et_fname,isfullscreen=isfullscreen,background_col=background_col)
    do_setup(el_tracker)

else: # define psychopy window
    win = visual.Window(color=background_col,colorSpace='rgb255',units='pix',checkTiming=True,fullscr=isfullscreen)

frameRate = 59.883 # hard coded
wWidth,wHeight= win.size
print(f"screen size is: {win.size}")

# participant will be looking at fixation cross
fixation_cross = visual.TextStim(win, text='+', color='white',height=50, units='pix')

# load audio file
snd = sound.Sound(soundFile) 
snd.setVolume(1)
durExp = snd.getDuration() # + 1  # Duration of experiment, in seconds
print(f"experimentDuration is {snd.getDuration()}\n\n")

# instruction screen(s)

# HEAD LOC INSTRUCTION
text_1 = 'Run ' + str(run)
text_1 += ': Head localization'
text_1 += '\n\n\nPlease close your eyes and rest for a minute.'
text_1 += '\n\n\n[start the recording and hit r to continue]'

draw_text(visual.TextStim(win,text_1),win)
instruction_continue(['r'])

text_2 = '[Ready to go? Press 1,2,3 or 4 to start!]'
draw_text(visual.TextStim(win,text_2),win)
instruction_continue(response_keys)

# mouse visibility
win.mouseVisible = False 

# define some handy timers
frameN = -1
#frameTolerance = 0.0001  # how close to onset before 'same' frame
min_frame = 18

# store frame rate of monitor if we can measure it
'''
print(frameRate)
if frameRate != None:
    frameDur = 1.0 / round(frameRate)
    print(f'frameRate is {frameRate}')
else:
    print('framerate is 60 - hard coded')
    frameDur = 1.0 / 60.0  # could not measure, so guess

trigger_info['s_onset'] = list(trigger_info['t_onset'].to_numpy()*(1/frameDur))
'''
print(trigger_info)
frameN = -1  # Initialize frame count


# fixation_cross is kept ON during experiment
fixation_cross.draw()
win.flip() # does it work with win.flip()


# Initialize Psychopy Clock object
trialClock = core.Clock()

while trialClock.getTime() < durExp + 2:
    # Get current time from Psychopy Clock
    t = trialClock.getTime()

    if snd.status != STARTED:
        frameN = frameN+1

    # Start/stop SOUND FILE
    if snd.status == NOT_STARTED and frameN >= min_frame:
        print(f'started: {frameN}')
        # Keep track of start time/frame for later
        snd.frameNStart = frameN  # exact frame index
        snd.tStart = t  # local t and not account for scr refresh
        snd.play()  # sync with win flip
        if ismeg:
            win.callOnFlip(trigger, port=port, code=150)
        if iseyetracking:
            # Sync event to eyetracker
            el_tracker.sendMessage(f'Play Sound | t={t - snd.tStart}')
        snd.status = STARTED
        print(f"first time stamp: {t}")

    if snd.status == STARTED:
        # Is it time to stop? (based on global clock, using actual start)
        if t - snd.tStart >= durExp:
            print("I went in!")
            snd.tStop = t
            snd.frameNStop = frameN
            snd.stop()
            saveExpLog(snd, fname_data, quitPressed=0)
            if iseyetracking:
                el_tracker.sendMessage(f'Stop Sound | t={t - snd.tStart}')
            print(f'stopped: {frameN}, {t - snd.tStart}')
            break
        else:
            # mark onset of task
            filtered_rows = trigger_info[(trigger_info['t_onset'] <= (t - snd.tStart)) & (trigger_info['status_onset'] == 0)]
            if not filtered_rows.empty:
                idx = filtered_rows.index[0]
                trigger_info.loc[idx, 'status_onset'] = 1
                print(f'{trigger_info.iloc[idx].condition_name} onset:', trigger_info.iloc[idx].t_onset, t - snd.tStart)
                if iseyetracking:
                    el_tracker.sendMessage(f'condition{trigger_info.iloc[idx].condition_name}_onset | t={t - snd.tStart}')
            # @@ mark offset of task
            filtered_rows = trigger_info[(trigger_info['t_offset'] <= (t - snd.tStart)) & (trigger_info['status_offset'] == 0)]
            if not filtered_rows.empty:
                idx = filtered_rows.index[0]
                trigger_info.loc[idx, 'status_offset'] = 1
                print(f'{trigger_info.iloc[idx].condition_name} offset:', trigger_info.iloc[idx].t_offset, t - snd.tStart)
                if iseyetracking:
                    el_tracker.sendMessage(f'condition{trigger_info.iloc[idx].condition_name}_offset | t={t - snd.tStart}')
            
    # Check for user input or other conditions to end the experiment
    pressed = event.getKeys(keyList=response_keys, modifiers=False, timeStamped=False)
    if pressed == ['q']:
        exit_experiment()


# close eyetracking connection
if iseyetracking:
    exit(el_tracker,et_fname,results_folder=results_path)

# close
text = visual.TextStim(win,f"End of the run.\n")
text.draw()
win.flip()
core.wait(3)

text_3 = 'Running head localization.\n\n\nPlease close your eyes and rest for a minute.\n\n\n[hit c to close]'
draw_text(visual.TextStim(win,text_3),win)
instruction_continue(['c','q'])


print('saving info...')
saveExpLog(snd,fname_data,quitPressed=0)
win.mouseVisible = True
win.close()
