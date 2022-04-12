#!/usr/bin/env python3

import rospy, ros_numpy
import numpy as np
import pickle
import os
import json
import actionlib
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import csv
from pprint import pprint
import pandas as pd
import json
import dataset
from dataset import run,scene,obstacle
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
from cv_bridge import CvBridge
import shutil
import random

from sklearn.neighbors import NearestNeighbors

def read_scene_from_pickle(): 
    scene_loaded = []
    path_scene=os.path.join(os.path.dirname(__file__), 'data/scene/data_scene')
    if os.path.isfile(path_scene):
        with (open(path_scene, "rb")) as openfile:
            while True:
                try:
                    scene_loaded.append(pickle.load(openfile))
                except EOFError:
                    break
        openfile.close()
    # exit()
    return scene_loaded

def read_json(path_run,n_run):
    run_loaded = []

    for i in range(n_run): 
        file=path_run+'dataset_run_' +str(i)+'.json'
        with (open(file, "rb")) as openfile:
            run_loaded.append(json.load(openfile))
    return run_loaded

def last_run_number(path_run):
    a=os.listdir(path_run) #NB: non sono in ordine
    if a==[]:
        b=-1
    else:
        b=len(a)-1
    return b 

#STRUTTURA JSON FILE
# dict_to_write={'init_pose':run.init_pose,
#                'goal_pose':run.goal_pose,
#                'run_total_time':run.run_total_time,
#                'timestamp':run.timestamp,  # i file json non possono contenere dati di tipo Time 
#                'data_laser':run.data_laser,
#                'data_imu':run.data_imu,           
#                'data_pose':run.data_pose,
#                'data_camera_image_path':run.data_camera_image,
#                'data_mmwave':single_mmwave_read,
#                'data_camera_depth_path':run.data_camera_depth
#                }


def rot_z(x,y,theta): #prende un vettore xy e lo ruota attorno a z di un determinato theta
    r = np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))
    v = np.array((x,y))
    return r.dot(v)

def obs_fake_poincloud(n_point,SceneID_select):
    scene_load=read_scene_from_pickle()
    obs_point=[]
    dim=int(n_point/scene_load[SceneID_select].num_obstacle)
    for i in range (scene_load[SceneID_select].num_obstacle):
        if i==scene_load[SceneID_select].num_obstacle-1:
            dim=n_point-dim*i
        l1=scene_load[SceneID_select].obs[i].l1
        l2=scene_load[SceneID_select].obs[i].l2
        fake_obs_point=np.column_stack([np.random.uniform(-l1/2,l1/2,dim),np.random.uniform(-l2/2,l2/2,dim),np.random.uniform(0,0.25,dim)])
        for j,element in enumerate(fake_obs_point):
            x,y=rot_z(element[0],element[1],scene_load[SceneID_select].obs[i].data_pose_obj[2])
            fake_obs_point[j][0]=x+scene_load[SceneID_select].obs[i].data_pose_obj[0]
            fake_obs_point[j][1]=y+scene_load[SceneID_select].obs[i].data_pose_obj[1]
        if i==0:
            obs_point=fake_obs_point
        else:
            obs_point=np.vstack([obs_point,fake_obs_point])

    #plot dei soli ostacoli

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_ylim([0,6])
    # ax.set_xlim([0,6])
    # ax.set_zlim([0,2])
    # for element in obs_point:
    #     ax.scatter(element[:,0],element[:,1],element[:,2])  
    # plt.show()
    return obs_point

def chamfer_distance(point1, point2, metric='l2'):
    """
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.

    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(point1)
    min_y_to_x = x_nn.kneighbors(point2)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(point2)
    min_x_to_y = y_nn.kneighbors(point1)[0]
    chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
        
    return chamfer_dist


def plot_all_run(mmwave,pose,scene_selected):
    point_x=[]
    point_y=[]
    point_z=[]
    pose_x=[]
    pose_y=[]
    pose_z=[]
    for i,single_read in enumerate(mmwave):
        # pose[i]
        for x,y,z in single_read:
            v=rot_z(x,y,pose[i][2])
            point_x.append(v[0]+pose[i][0])  #riporto i punti nel sdr globale
            point_y.append(v[1]+pose[i][1])
            point_z.append(z)
            pose_x.append(pose[i][0])
            pose_y.append(pose[i][1])
            pose_z.append(0.1)
    
    # tolgo i punti fuori dall'area di visione del vicon 
    #----------------------------------------------------------------------------
    index=[]
    for i in range(len(point_x)):
        if point_x[i]<0 :
            index.insert(-1, i)
        elif point_y[i]<0 or point_y[i]>5:
            index.insert(-1, i)
        elif point_z[i]<0 :
                index.insert(-1, i)
    point_x = np.delete(point_x, index)
    point_y = np.delete(point_y, index)
    point_z = np.delete(point_z, index)
    #------------------------------------------------------------------------
    subset_size = int(0.80 * len(point_x)) #estraggo un subset dei vettori di punti per il plot
    point_x=np.random.choice(point_x, subset_size, replace=False)
    point_y=np.random.choice(point_y, subset_size, replace=False)
    point_z=np.random.choice(point_z, subset_size, replace=False)
    # pose_x=np.random.choice(pose_x, subset_size, replace=False)
    # pose_y=np.random.choice(pose_y, subset_size, replace=False)
    # pose_z=np.random.choice(pose_z, subset_size, replace=False)
    n_point=len(point_x)
    obs_point=obs_fake_poincloud(n_point,scene_selected) #crea nuvole punti false 
    point_run=np.column_stack([point_x,point_y,point_z])
    cd=chamfer_distance(point_run, obs_point)
    print("chamfer distance= ",cd)



    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(point_x,point_y,point_z)
    ax.scatter(pose_x,pose_y,pose_z,color='orange')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(obs_point[:,0],obs_point[:,1],obs_point[:,2],color='red')
    # ax.set_xlim([0,max_depth])
    # # ax.set_ylim([-max_sezione_y,max_sezione_y])
    # # ax.set_zlim([-max_sezione_z,max_sezione_z])
    ax.set_ylim([0,6])
    ax.set_xlim([0,6])
    ax.set_zlim([0,2])
    plt.show()



def main():
    
    path_run=os.path.join(os.path.dirname(__file__), 'data/json_conversion/')
    n_run=last_run_number(path_run)+1 #last_run_number ritorno l'ID della run (che parte da 0, quindi con 3 run ritorna 2)
    run_loaded=read_json(path_run,n_run) #restituisce una lista con tutte le info di tutte le run 
    #Ad esempio: run_loaded[0] sarà un dict in cui tutti gli elementi (sensori,pose,ecc..) della prima run 
    print("Numero run= ",len(run_loaded))
    run_selected=37
    scene_selected=run_loaded[run_selected]['scene_ID']
    mmwave=run_loaded[run_selected]['data_mmwave'] #contiene tutte le letture del mmwave della run selezionata

    pose_init=run_loaded[run_selected]['init_pose']
    pose=run_loaded[run_selected]['data_pose']
    pose_relative= [np.array(pose[i][0:2])-np.array(pose_init[0:2]) for i in range(len(pose))]
    plot_all_run(mmwave,pose,scene_selected)



if __name__ == '__main__':
    main()

