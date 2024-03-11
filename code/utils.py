import shutil
import os
import statistics 
import scipy.stats as stats
import math
from itertools import cycle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


COLSETS = {'eyes':['AU01_r', 'AU02_r', 'AU04_r'], 'midpart':['AU05_r', 'AU06_r', 'AU07_r', 'AU09_r'],
    'mouth':['AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'], 
    'all':['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']}

def get_filenames(directory_name:str, target_file:str='frames') -> list:
    '''
    walks through a given directory and collects all files that contain the word file
    input: directory_name, file - a word that bust be included in the file name
    '''
    return [file for file in os.listdir(directory_name) if target_file in file]


def new_folder(fold_path:str) -> None:
    '''
    creates a new folder on the path 'fold_path'
    '''
    if os.path.exists(fold_path):
        shutil.rmtree(fold_path)
    os.makedirs(fold_path)



################  CUTS  ########################


#inner function
#input: a pandas dataframe which represent one shot
#return: number of cuts; duration of each of them
#how to find a shot: 
# shots = df.face_id.unique() #find unique face_ids
# for id in shots:
#   shot = df[df['face_id'] == id]

def find_cats(shot):
    
    output = []
    shot = shot.reset_index()
    length = len(shot)
    init_frame = shot.iloc[0]['frame']
    current = init_frame
    init_time = shot.iloc[0]['timestamp']

    for i in range(length):
        temp = shot.iloc[i] 

        if temp['frame'] != current: #counting continous sequence of the frames
            output.append([init_frame, shot.iloc[i-1]['frame'], shot.iloc[i-1]['timestamp'] - init_time])
            init_frame = temp['frame']
            current = init_frame
            init_time = temp['timestamp']
        current += 1

    output.append([init_frame, shot.iloc[i]['frame'], shot.iloc[i]['timestamp'] - init_time])

    return output

#the function check if there are several cuts in one shot, and rename the shots
#it numerates cuts in the way: 27.0, 27.1, 27.2... or assign -1 to cuts lesser than 3sec
#output: dataframe with renamed face_id fields, number of cuts

def cuts_split(_df):
    df = _df
    shots = df.face_id.unique() #find unique face_ids
    cuts = 0
    cuts_logs = []

    for id in shots:
        shot = df[df['face_id'] == id]
        temp = find_cats(shot)

        if len(temp) > 1: #if there are cuts in the shot
            cuts += 1 #iterate the amount of shots with cuts
            print('The face_id that is changed:', id, temp, len(temp))
            cuts_logs.append(str(id) + ' ' + str(temp) + ' ' + str(len(temp)))
            for j in range(len(temp)):
                if temp[j][2] >= 3.0: #if a current cut is longer than 3 sec
                    df.loc[(df['frame'] >= temp[j][0]) & (df['frame'] <= temp[j][1]) & (df['face_id'] == id), 'face_id'] = id + j/10 #change face_id
                    #df[(df['frame'] >= temp[j][0]) & (df['frame'] <= temp[j][0])]['face_id'] = id + j/10 #change face_id
                else:
                    #df[(df['frame'] >= temp[j][0]) & (df['frame'] <= temp[j][0])]['face_id'] = -1 #mark as a negative number to delete later
                    df.loc[(df['frame'] >= temp[j][0]) & (df['frame'] <= temp[j][1]) & (df['face_id'] == id), 'face_id'] = -1 #mark as a negative number to delete later

    print('The amount of shots with cuts:', cuts)
    
    return df, cuts_logs




def cuts_split_20percent(_df, frame_height):
    df = _df
    shots = df.face_id.unique() #find unique face_ids
    cuts = 0
    cuts_logs = []

    id_reset = 0

    for id in shots:
        shot = df[df['face_id'] == id]

        ytop = list(shot['y_27'])[0]
        ybott = list(shot['y_8'])[0]

        face_percentage = abs(ytop - ybott) * 1.6 > 0.2 * frame_height
        #print(id_reset, 'face:', abs(ytop - ybott) * 1.6,  'frame 20 percent:', 0.2 * frame_height, face_percentage)


        temp = find_cats(shot)

        if len(temp) > 1: #if there are cuts in the shot
            cuts += 1 #iterate the amount of shots with cuts
            #print('The face_id that is changed:', id, temp, len(temp))
            cuts_logs.append(str(id) + ' ' + str(temp) + ' ' + str(len(temp)))

            for j in range(len(temp)):
                if temp[j][2] >= 3.0 and face_percentage: #if a current cut is longer than 3 sec
                    df.loc[(df['frame'] >= temp[j][0]) & (df['frame'] <= temp[j][1]) & (df['face_id'] == id), 'face_id'] = id_reset #change face_id
                    id_reset += 1
                else:
                    df.loc[(df['frame'] >= temp[j][0]) & (df['frame'] <= temp[j][1]) & (df['face_id'] == id), 'face_id'] = -1 #mark as a negative number to delete later
        
        elif not face_percentage: # if there are no cuts
            df.loc[df['face_id'] == id, 'face_id'] = -1
            #print('deleted face: ', id)
        else:
            df.loc[df['face_id'] == id, 'face_id'] = id_reset
            id_reset += 1


    print('The amount of shots with cuts:', cuts)
    print(id_reset)
    
    return df, cuts_logs


#transforms time in ssss.msmsms format to hh:mm:ss:ms
def time_transform(timestamp):

    time = timestamp.split('.')
    if len(time[1]) ==1:
        time[1] += '00'
    if len(time[1]) ==2:
        time[1] += '0'

    trial_t = int(time[0]) * 1000 + int(time[1])
    #print(trial_t)
    
    trial_h = (trial_t//3600000)%100 #in case an amount of hours is more than 99 hours
    trial_min = (trial_t % 3600000)//60000
    trial_s = (trial_t % 60000) //1000
    trial_ms = trial_t % 1000
    if trial_h > 9:
        time = str(trial_h) + ':'
    else:
        time = '0' + str(trial_h) + ':'
    if trial_min > 9:
        time = time + str(trial_min)
    else:
        time = time + '0' + str(trial_min)
    if trial_s > 9:
        time = time + ':' + str(trial_s)
    else:
        time = time + ':' + '0' + str(trial_s)
    if trial_ms > 0:
        time = time + '.' + str(trial_ms)
    else:
        time = time + '.' + '000'
    
    return time