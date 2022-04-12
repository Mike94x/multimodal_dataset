#!/usr/bin/env python3
from cmath import sin

from matplotlib.pyplot import axes
import rospy, ros_numpy
import numpy as np
import pickle
import os
import json
import actionlib
from PIL import Image
import os
import datetime
import time
import csv
from pprint import pprint
import pandas as pd
import json
import dataset
from dataset import run,scene,obstacle
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# funzione che carica tutte le run presenti
def read_from_pickle(path_run,n_run):
    run_loaded = []
    for i in range(n_run+1): 
        file=path_run+"run"+str(i)+"/data_run"+str(i)
        print(file)
        # exit()
        if os.path.isfile(file):
            with (open(file, "rb")) as openfile:
                while True:
                    try:
                        run_loaded.append(pickle.load(openfile))
                        # print("ID run= ",run_loaded[i].ID_run)
                        # exit()
                    except EOFError:
                        break
            openfile.close()

    return run_loaded

def last_run_number(path_run):
    a=os.listdir(path_run) #NB: non sono in ordine, spesso sono mischiati!!!!
    if a==[]:
        b=-1
    else:
        # b=max(a)[3:len(max(a))] #l'ultimo id run sarà il max di a dal carattere 3 in poi (al netto della parola "run")
        b=len(a)-1
    return b 

def convert_dataset_to_json(run):
    single_mmwave_read=[]
    for item in run.data_mmwave:
        if len(item)>1:
            item=[np.vstack((item[0],item[1]))]
        # print(len(item))
        # print(len(item))
        single_mmwave_read = [*single_mmwave_read,*item] 
    
    dict_to_write={'scene_ID':run.s_ID,
                   'init_pose':run.init_pose,
                #    'goal_pose':run.goal_pose,
                   'run_total_time':run.run_total_time,
                   'timestamp':run.timestamp,  
                   'data_laser':run.data_laser,
                   'data_imu':run.data_imu,           
                   'data_pose':run.data_pose,
                   'data_camera_image_path':run.data_camera_image,
                   'data_mmwave':single_mmwave_read,   
                   'data_camera_depth_path':run.data_camera_depth
                   }

    print("converto run",run.ID_run)
    path_to_file=os.path.join(os.path.dirname(__file__), 'data/json_conversion/')
    # path_to_file=os.path.join(os.path.dirname(__file__), 'data/json_conversion_manual/')
    path_to_file+="dataset_run_"+str(run.ID_run)+".json"
    with open(path_to_file,'w') as json_file:
        json.dump(dict_to_write,json_file,cls=NumpyArrayEncoder,indent=4) #,default=str)


def main():
    path_run=os.path.join(os.path.dirname(__file__), 'data/run/')
    filename=os.path.join(os.path.dirname(__file__), 'data/scene/') #nome/percorso del file che contiene le scene salvate
    # path_run=os.path.join(os.path.dirname(__file__), 'data/run_manual/')
    # filename=os.path.join(os.path.dirname(__file__), 'data/scene_manual/') #nome/percorso del file che contiene le scene salvate
    n_run=last_run_number(path_run)+1 #last_run_number ritorno l'ID della run (che parte da 0, quindi con 3 run ritorna 2)
    run_loaded=read_from_pickle(path_run,n_run)
    print("Numero run= ",len(run_loaded))
    for r in run_loaded: 
        convert_dataset_to_json(r)



if __name__ == '__main__':
    main()