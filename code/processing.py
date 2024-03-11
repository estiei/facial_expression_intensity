import pandas as pd

def coord_match(dset1:pd.DataFrame, dset2:pd.DataFrame) -> bool:
    '''
    get a vector of coordinates of two faces - dset1 and dset2
    if False, the face in a folowing frame is too far from the face in a current dataset
    '''

    match = True
    width = abs(dset1[0] - dset1[1]) / 2
    height = abs(dset1[-1] - dset1[-2]) / 2
    for i in range(len(dset1) // 2):
        if (abs(dset1[i] - dset2[i]) > width) or (abs(dset1[i + len(dset1) // 2] - dset2[i + len(dset1)//2]) > height):
            match = False
    return match


def distance(dset1:list, dset2:list) -> float:
    '''
    get a vector of coordinates of two faces - dset1 and dset2 
    returns the sum of distances between related dots (e.g., x_8 from dset1 and x_8 from dset2)
    '''

    distance = 0
    
    for i in range(len(dset1)):
        distance += abs(dset1[i] - dset2[i])
    
    return distance


def face_fitness(dset):
    time_start = float(dset.iloc[0]['timestamp']) #first row
    time_end = float(dset.iloc[-1]['timestamp']) #last row

    if time_end - time_start < 3.0:
        # print(time_end - time_start)
        return False
    
    success = list(dset['success'])

    return True


def compare(frame1:dict, frame2:dict) -> tuple[list, list, dict]:
    '''
    This function compares two lines of a dataset: current line (frame2) and previous line (frame1).
    For each face we look for a match between frame1 and frame2 in order to track faces continuously.
    Faces are compared based on their 2d landmarks.

    Function returns a list with unmatched faces (to_delete), new faces (to add), matched faces.
    '''

    col_of_int = ['x_0', 'x_16', 'x_8', 'x_27', 'y_0', 'y_16', 'y_8', 'y_27']
    #calculate distances
    matches = {}
    to_delete = [] # id of datasets presented in frame1, but don't exist anymore in frame2
    to_add = [] # id of datasets presented in frame2, but absence in frame1

    arbit_key = list(frame2.keys())[0]
    
    for key1 in frame1: # walk trough dset1
        matched = -1
        #add = -1
        last_line1 = frame1[key1].iloc[-1] # take the last row from a dataset
        last_line2 = frame2[arbit_key].iloc[-1] # take the last row from a dataset
        min_dist = distance(last_line1[col_of_int], last_line2[col_of_int])

        for key2 in frame2:
            last_line1 = frame1[key1].iloc[-1]
            last_line2 = frame2[key2].iloc[-1]
            temp = distance(last_line1[col_of_int], last_line2[col_of_int])
            
            if temp <= min_dist and coord_match(last_line1[col_of_int], last_line2[col_of_int]): #dataset in presented in frame2
                min_dist = temp
                matched = key2 #then add this J dataset to I dataset from frame1
        if matched != -1:
            matches[key1] = matched
        
    #print(matches)
    keys1 = list(frame1.keys())
    keys2 = list(frame2.keys())

    matched_f1 = list(matches.keys())
    matched_f2 = list(matches.values())

    for i in keys1:
        if i not in matched_f1:
            to_delete.append(i)
    
    for i in keys2:
        if i not in matched_f2:
            to_add.append(i)
        
    return to_delete, to_add, matches