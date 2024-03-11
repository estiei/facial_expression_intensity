import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

from processing import * 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--film_name')
    parser.add_argument('--input_folder', default='./dataset/12movies_init_FaceReader_output')
    parser.add_argument('--output_folder', default='./data_intermediate/12movies_selected_frames')
    parser.add_argument('--lower_time_limit', default=3)
    parser.add_argument('--upper_time_limit', default=5)

    args = parser.parse_args()

    df = pd.read_csv(f'{args.input_folder}/{args.film_name}.csv', delimiter=',')
    cols = df.columns

    col_of_int = ['x_0', 'x_16', 'x_8', 'x_27', 'y_0', 'y_16', 'y_8', 'y_27']

    count = 0
    last_faceid = 0 # face in the final dataset
    faceid = 0 # last face in the current_frame (tracker)
    faces = {} # dict of datasets

    # cycle for the first frame data extraction. It will then be compared with the next frames
    current_frame = {}
    single_line = df.iloc[count]
    frame = single_line['frame']

    # initial dataset consists of faces presented in the first frame
    while True:
        detected_faces = df[count:count+1]
        single_line = df.iloc[count]
        if single_line['frame'] != frame:
            break
        current_frame[faceid] = detected_faces
        faceid += 1 # last faceID in the 
        count += 1

    # cycle for data analysis
    # take another frame
    pbar = tqdm(total = len(df))
    while count < len(df)-1:  
        subsequent_frame = {}
        single_line = df.iloc[count]
        frame = int(single_line['frame'])

        while True:
            detected_faces = df[count:count+1]
            single_line = df.iloc[count]
            id = int(single_line['face_id'])
            if single_line['frame'] != frame:
                break
            subsequent_frame[id] = detected_faces
            count += 1


        delete, add, matches = compare(current_frame, subsequent_frame) #compare faces from the previous and the current frames

        # add data to current_frame if a new face appears
        for i in add:
            current_frame[faceid] = subsequent_frame[i]
            faceid += 1

        # face do not exist anymore in the new dataset
        # remove it from current_frame; do checkig - if fits, add it to the final vocab (faces)
        for i in delete:
            if face_fitness(current_frame[i]) == True:
                faces[last_faceid] = current_frame[i]
                last_faceid += 1
            current_frame.pop(i, None)
        
        for i in matches:
            current_frame[i] = pd.concat([current_frame[i], subsequent_frame[matches[i]]])
        pbar.update(count - pbar.n)
    pbar.close()
        
    for i in tqdm(current_frame):
        if face_fitness(current_frame[i]) == True:
            faces[last_faceid] = current_frame[i]
            last_faceid += 1

    result_faces = {}
    timestamps = []
    face_id = 0
    newkey = 0
    total_time = 0

    id = 0
    for key in tqdm(faces):
        data = faces[key]

        temp = data.iloc[0]
        time_start = temp['timestamp']
        frame_start = temp['frame']
        frame1 = 0
        time_end = temp['timestamp']
        frame_end = temp['frame']
        frame2 = 0

        count = 0

        new_dset = pd.DataFrame(columns=cols)
        flag = False

        while count < len(data)-1:
            temp = data.iloc[count]
            success = int(temp['success'])
            if success == 0:
                count += 1

                if count >= len(data)-1:
                    break
                continue

            time_start = float(temp['timestamp'])
            frame_start = temp['frame']
            frame1 = count
            while success == 1:
                temp = data.iloc[count]
                time_end = float(temp['timestamp'])
                frame_end = temp['frame']
                frame2 = count
                count += 1
                if count >= len(data)-1:
                    break
                success = int(temp['success'])
            
            # select frames longer than lower_time_limit and shorter than upper_time_limit
            if time_end - time_start > args.lower_time_limit and time_end - time_start < args.upper_time_limit:
                total_time += time_end - time_start
                new_dset = pd.concat([new_dset, faces[key][frame1:frame2]])
                flag = True
                #fram_end - is the next not suitable frame, so the previous is taken
                timestamps.append([face_id, time_start, time_end, frame_start, frame_end-1])
            
        
        if flag:
            result_faces[newkey] = new_dset
            newkey += 1
            face_id += 1

    output = pd.DataFrame(columns=cols)

    for i in tqdm(result_faces):
        result_faces[i]['face_id'].values[:] = i
        output = pd.concat([output, result_faces[i]])

    
    #save the timestamps file with the selected frames
    output = pd.DataFrame(np.array(timestamps),
                          columns=['face_id', 'time_start', 'time_end', 'frame_start', 'frame_end'])

    pd.DataFrame.from_dict(output).to_csv(f'{args.output_folder}/timestamps_{args.film_name}.csv', sep=',', mode='w')