#!/usr/bin/env python3
# from re import A
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


bridge = CvBridge()
img_size=[480,640]
# CAMERA INFO   
horizontal_ang_camera=37 #37° = ha un cono di 74 totale (da -37 a +37)
vertical_ang_camera=31
step_angol_per_pixel_vert=vertical_ang_camera/(img_size[0]/2) #ogni pixel vericale corrisponde a questo valore di angolo
step_angol_per_pixel_horiz=horizontal_ang_camera/(img_size[1]/2) #ogni pixel orizzontale corrisponde a questo valore di angolo
fl=1.93 #focal lenght: possiamo definirla come la distanza tra il piano dell'immagine e la camera
sez_piano_focale_x= fl*np.tan(np.deg2rad(horizontal_ang_camera)) #mezza dimensione del piano x a distanza pari a fl
sez_piano_focale_y= fl*np.tan(np.deg2rad(vertical_ang_camera)) #mezza dimensione del piano y a distanza pari a fl
# Intrinsic camera matrix for the raw (distorted) images.
#     [fx  0 cx]
# K = [ 0 fy cy]
#     [ 0  0  1]
# K=[614.636962890625, 0.0, 325.5340881347656, 0.0, 614.7501831054688, 251.1617889404297, 0.0, 0.0, 1.0]
# Projects 3D points in the camera coordinate frame to 2D pixel
# coordinates using the focal lengths (fx, fy) and principal point
# (cx, cy).

# MMWAVE INFO   
horizontal_ang_mmWave= 60 #60°
vertical_ang_mmWave=15 #15°
max_depth=4.2
max_sezione_y=max_depth*np.tan(np.deg2rad(horizontal_ang_mmWave)) 
max_sezione_z=max_depth*np.tan(np.deg2rad(vertical_ang_mmWave)) 

count_total_data=0 #usata per dare un numero crescente alle depthmap e alle img per crerare il database totale per la rete

def open_image(path_img):
    return cv2.imread(path_img,cv2.IMREAD_UNCHANGED)

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

def create_rgbd(run):
    mmwave_agg=[]
    depth_map=[]
    global count_total_data
    img_color_path=run['data_camera_image_path']
    img_depth_path=run['data_camera_depth_path']
    mmwave=run['data_mmwave']
    mmwave_agg=[[*mmwave_agg,*mmwave[i]] for i in range(len(mmwave))]

    for i in range(len(mmwave_agg)): # sono tutte le nuvole punti di 1 run
        # pixel=[]
        x = np.array([x for x, y, z in mmwave_agg[i]])
        y = np.array([y for x, y, z in mmwave_agg[i]])
        z = np.array([z for x, y, z in mmwave_agg[i]])
        dist_x_camera_plane= (y-0.1/x)*fl #distanza del pixel sul piano x dell'immagine
        dist_y_camera_plane= ((z)/x)*fl #distanza del pixel sul piano y dell'immagine
        px=(img_size[1]/2)*dist_x_camera_plane/sez_piano_focale_x #proporizione=   320pixel:sez_piano_focale_x=px:dist_x_camera_plane
        py=(img_size[0]/2)*(dist_y_camera_plane)/sez_piano_focale_y # proporizione=   240pixel:sez_piano_focale_y=py:dist_y_camera_plane
        
        #correggo in modo tale da non avere l'indice 0,0 al centro della matrice (indici solo positivi)
        px=(px+img_size[1]/2).astype(int)
        py=(py+img_size[0]/2).astype(int)
        
        #correzione di valori letti dal mmwave che sono fuori dal cono di lettura
        index=[]
        for k in range(len(px)):
            if px[k]>img_size[1] or px[k]<0:
                index.insert(-1, k)
            elif py[k]>img_size[0] or py[k]<0:
                index.insert(-1, k)
        px = np.delete(px, index)
        py = np.delete(py, index)
        px-=1 # -1 perchè i pixel vanno da 0 a 599
        py-=1 #-1 perchè i pixel vanno da 0 a 479
        x = np.delete(x, index)
        y = np.delete(y, index)
        z = np.delete(z, index)
        depth=np.ones((img_size[0],img_size[1]))*(-1)
        depth[py,px]=x
        depth=np.flip(depth) 
        depth_map.append(depth)
        file_depth=os.path.dirname(__file__)+'/dataset_network/deptmap/depthmap'+str(count_total_data)+'.png'
        print(file_depth)
        file_groundt=os.path.dirname(__file__)+'/dataset_network/groundtruth/groundtruth'+str(count_total_data)+'.png'
        save_image(file_depth,depth) #salva le immagini (path_to_file,immagine,formato)
        shutil.copy2(img_depth_path[i], file_groundt) # complete target filename given
        file_image=os.path.dirname(__file__)+'/dataset_network/image/image'+str(count_total_data)+'.png'
        shutil.copy2(img_color_path[i], file_image) # complete target filename given
        count_total_data+=1
        
        # exit()
    return depth_map


def save_image(path,image):
    image = np.clip(image * 65535, 0, 65535) # proper [0..255] range
    image = image.astype(np.uint16)  # safe conversion
    cv2.imwrite(path, image)

def main():
    path_run=os.path.join(os.path.dirname(__file__), 'data/json_conversion/')
    # path_run=os.path.join(os.path.dirname(__file__), 'data/json_conversion_manual/')
    path_scene=os.path.join(os.path.dirname(__file__), 'data/scene/') #nome/percorso del file che contiene le scene salvate
    n_run=last_run_number(path_run)+1 #last_run_number ritorno l'ID della run (che parte da 0, quindi con 3 run ritorna 2)
    print(n_run)
    run_loaded=read_json(path_run,n_run) #restituisce una lista con tutte le info di tutte le run 
    #Ad esempio: run_loaded[0] sarà un dict in cui tutti gli elementi (sensori,pose,ecc..) della prima run 
    print("Numero run= ",len(run_loaded))
    run_selected=37
    frame_selected=63
    
    mmwave=run_loaded[run_selected]['data_mmwave']#[frame_selected] #contiene la lettura 10 del mmwave della run 1

    pose_init=run_loaded[run_selected]['init_pose']
    pose=run_loaded[run_selected]['data_pose']
    img_path=run_loaded[run_selected]['data_camera_image_path'][frame_selected]
    depth_img_path=run_loaded[run_selected]['data_camera_depth_path'][frame_selected]
    pose_relative= [np.array(pose[i][0:2])-np.array(pose_init[0:2]) for i in range(len(pose))]


    depth_map=[]
    for r in run_loaded:
        depth_map.append(create_rgbd(r))



    # la parte successiva serve solo per sovrapporre l'immagine con la depthmap per vedere visivamente se sono sensate
    
    # mmwave[frame_selected]
    # x = np.array([x for x, y, z in mmwave[frame_selected]])
    # y = np.array([y for x, y, z in mmwave[frame_selected]])
    # z = np.array([z for x, y, z in mmwave[frame_selected]])
    # img=open_image(img_path)
    # img2=img


    # depth_to_plot=depth_map[run_selected][frame_selected]
    # img2[np.where(depth_to_plot!=-1)]=255 #pongo i pixel che hanno un depth diversa da -1 a bianco (per verificare solo se la posizione è corretta, non serve come dato)
    

    # plt.figure()
    # plt.imshow(img2)
    # plt.figure()
    # plt.imshow(depth_to_plot)
    

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x,y,z)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim([0,max_depth])
    # # ax.set_ylim([-max_sezione_y,max_sezione_y])
    # # ax.set_zlim([-max_sezione_z,max_sezione_z])
    # ax.set_ylim([-3.16,3.16])
    # ax.set_zlim([-2.52,2.52])
    # plt.show()




if __name__ == '__main__':
    main()