#!/usr/bin/env python3

from dis import code_info
import threading
from numpy.core.fromnumeric import shape

from numpy.core.shape_base import block
from nav_msgs.msg import Odometry
import rospy, ros_numpy
from geometry_msgs.msg import *
from turtle.msg import SafeVicon
import numpy as np
from rospy.timer import sleep
from tf.transformations import euler_from_quaternion,quaternion_from_euler
import sensor_msgs.msg as sm
import pickle
import os
import matplotlib.pyplot as plt
import json
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from math import radians, degrees
from actionlib_msgs.msg import *
from PIL import Image
import os
import cv2
# from matplotlib.patches import Polygon
import matplotlib.patches as patches
import roslaunch
import time
from cv_bridge import CvBridge

vic_x=3 #mezza dimensione vicon
vic_y=2.7
data_mmwave_last_value=[]
lock=threading.Lock()
bridge = CvBridge()
long_obs=''
obs_topic={ 0:'/vicon/Obstacle0/Obstacle0',
            1:'/vicon/Obstacle1/Obstacle1',
            2:'/vicon/Obstacle2/Obstacle2',
            3:'/vicon/Obstacle3/Obstacle3',
            4:'/vicon/LongObstacle01/LongObstacle01',
            5:'/vicon/LongObstacle23/LongObstacle23'}

def get_angle(quaternion_pose):
    q = [quaternion_pose.x,
            quaternion_pose.y,
            quaternion_pose.z,
            quaternion_pose.w]
    roll, pitch, yaw = euler_from_quaternion(q)
    return yaw

def rot_z(x,y,theta):
    r = np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))
    v = np.array((x,y))
    return r.dot(v)

def save_image(path,image,format):
    cv2.imwrite(path,bridge.imgmsg_to_cv2(image, format))

class obstacle():
    # data_pose_obj=[]
    # h=0
    # r=0
    # ID_obs=0
    def __init__(self,l1,l2,i,obs_id):
        global obs_topic
        #i indica il numero dell'ostacolo
        self.data_pose_obj=0
        # obj='Obstacle'+str(i)
        # self.topic='/vicon/'+str(obj)+'/'+str(obj)
        self.topic=obs_topic[obs_id[i]]
        self.ID_obs=obs_id[i]
        if  obs_topic[obs_id[i]][0:19]=='/vicon/LongObstacle':
            self.l1=l1 # dimensione lungo l'asse x (quella più piccola nel nostro caso)
            self.l2=l2*4 # dimensione lungo l'asse y
        else:
            self.l1=l1 # dimensione lungo l'asse x (quella più piccola nel nostro caso)
            self.l2=l2 # dimensione lungo l'asse y

        thread_o = threading.Thread(target = self.vicon_odom)
        thread_o.setDaemon(True)
        thread_o.start()
        # self.vicon_obj
        # print(self.topic)
        # print(self.ID_obs)
        rospy.sleep(0.1)
        
        
        
        # print("posa oggetto="+str(self.data_pose_obj))


    #funzione per fare il print degli attributi della classe (print(obstacle))
    def __str__(self):
        return str(self.__class__)+":"+str(self.__dict__)

    def vicon_odom(self):
        rospy.Subscriber(self.topic, SafeVicon, self.vicon_odomCD)
        rospy.spin()


    def vicon_odomCD(self,msg):
        data_pose_x = round(msg.transform.translation.x,4)+vic_x
        data_pose_y = round(msg.transform.translation.y,4)+vic_y
        theta = round(get_angle(msg.transform.rotation),4) 
        self.data_pose_obj=[data_pose_x, data_pose_y, theta]

 
class scene():
    # obs=[]
    # ID_scene=[]
    # k indica l'ID della scena attuale (lastID+1), scene_l contiene tutte le scene caricate già presenti nel file pickle (derivanti da def read_scene())
    def __init__(self,k,obs_id): #costruendo nuova scena
        self.obs=[]
        self.ID_scene=k
        self.num_obstacle=len(obs_id)
        #k indica il numero della scena (ID)
        for i in range (0,self.num_obstacle):
            l1=0.36 #input('Inserisci dimensione 1 ostacolo '+str(i)+':') #dimensione lungo x
            l2=0.49#input('Inserisci dimensione 2 ostacolo '+str(i)+':') #dimensione lungo y
            self.obs.append(obstacle(l1,l2,i,obs_id))
            print(self.obs[i].data_pose_obj)
            # print("theta ostacolo = ",str(np.rad2deg(self.obs[i].data_pose_obj[2])) +" gradi")
            # print("IN costruttore scena, ostacolo "+str(i)+"= "+str(self.obs[i]))
    #funzione per fare il print degli attributi della classe (print(scene))
    def __str__(self):
        return str(self.__class__)+":"+str(self.__dict__)



class run():
    
    # init_pose=[]
    # goal_pose=[]
    # ID_run=0
    # data_camera_image=[]
    # data_pose=[]
    # data_mmwave=[]
    # data_mmwave_last_value=[]
    # data_laser=[]
    # data_imu=[]
    # data_camera_depth=[]
    # s_ID=0 #conterrà l'ID della scena selezionata
    # scene_selected=[] #conterrà la scena selezionata
    def __init__(self,j,scene_select_ID,s,n_obs,obs_id):
        #j indica è l'identificativo (numero) della run, scene_select indica la scena selezionata (oggetto classe scena)
        self.ID_run=j
        self.s_ID=scene_select_ID #mi salvo l'ID della scena selezionata come attributo nell'oggetto run che sto creando
        self.scene_selected=s
        self.data_camera_image=[]
        self.data_pose=[]
        self.data_mmwave=[]
        self.data_laser=[]
        self.data_imu=[]
        self.data_camera_depth=[]
        self.timestamp=[]
        #ricreo gli ostacoli nella scena selezionata (senza modificare le scene salvate nel file) in modo tale da avere le posizione 
        #precisa del vicon (al netto delle tolleranze)
        for i in range (0,n_obs): 
            self.scene_selected.obs[i]=obstacle(self.scene_selected.obs[i].l1,self.scene_selected.obs[i].l2,i,obs_id)
        thread1 = threading.Thread(target = self.vicon_odom)
        thread2 = threading.Thread(target = self.camera_image)
        thread3 = threading.Thread(target = self.camera_depth)
        thread4 = threading.Thread(target = self.lidar)
        thread5 = threading.Thread(target = self.imu6dof)
        thread6 = threading.Thread(target = self.mmWave)
        thread1.setDaemon(True)
        thread1.start()
        thread2.setDaemon(True)
        thread2.start()
        thread3.setDaemon(True)
        thread3.start()
        thread4.setDaemon(True)
        thread4.start()
        thread5.setDaemon(True)
        thread5.start()
        thread6.setDaemon(True)
        thread6.start()

        rospy.sleep(4)
        self.init_pose=self.data_pose_last_value   
        self.goal_pose=self.random_goal_pose(n_obs)  #funzione con calcolo randomico goal e controllo su intersezione ostacolo
        print("self.init_pose= ",self.init_pose)
        print("self.goal_pose= ",self.goal_pose)
        self.moveToGoal() #prende (da self) la posizone finale finale calcolata per il navigation stack per fare la navigazione

    #funzione per fare il print degli attributi della classe (print(run))
    def __str__(self):
        return str(self.__class__)+":"+str(self.__dict__)

    #posa del turtlebot
    def vicon_odom(self): 
        rospy.Subscriber('/vicon/turtlebot3/turtlebot3', SafeVicon, self.vicon_odomCB) 
        rospy.spin()
    
    def vicon_odomCB(self,msg):
        data_pose_x = round(msg.transform.translation.x,4)+vic_x
        data_pose_y = round(msg.transform.translation.y,4)+vic_y
        theta = round(get_angle(msg.transform.rotation),4) 
        self.data_pose_last_value=[data_pose_x, data_pose_y, theta]

    def lidar(self): 
        rospy.Subscriber('/turtlebot3/scan', sm.LaserScan, self.lidarCB)
        rospy.spin()
    
    def lidarCB(self,msg):
        self.data_laser_last_value=msg.ranges

    def imu6dof(self): 
        rospy.Subscriber('/turtlebot3/imu', sm.Imu, self.imuCB)
        rospy.spin()

    def imuCB(self,msg):
        data_imu_w=np.array([msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z])
        data_imu_l_acc=np.array([msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z])
        self.data_imu_last_value=np.concatenate((data_imu_w, data_imu_l_acc), axis=None) #restituisce un array di 6 elementi (3 per vel angolare, 3 per acc lineare)
    
    def camera_image(self): 
        rospy.Subscriber('/turtlebot3/camera/color/image_raw', sm.Image, self.camera_imageCB)
        rospy.spin()

    def camera_imageCB(self,msg):
        self.data_camera_image_last_value=msg

    def camera_depth(self): 
        rospy.Subscriber('/turtlebot3/camera/depth/image_rect_raw', sm.Image, self.camera_depthCD)
        rospy.spin()
    
    def camera_depthCD(self,msg):
        self.data_camera_depth_last_value=msg

    def mmWave(self): 
        rospy.Subscriber('/ti_mmwave/radar_scan_pcl', sm.PointCloud2, self.mmWaveCB)
        rospy.spin()

    def mmWaveCB(self,msg):
        global data_mmwave_last_value
        global lock
        with lock:
            data_mmwave_last_value.append(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)) # pointcloud2_to_xyz_array mi restituisce una lista di punti (lista di liste xyz)
            # data_mmwave_last_value conterrà una lista di liste di punti (lista di liste di liste xyz)

    def random_goal_pose(self,n_obs):
        tol=0.2 #evito gi generare punto sul perimetro esterno del vicon perchè potrebbe perdere l'odometria (vicon non vede il robot)
        goal_x=np.random.uniform(0+tol,2*vic_x-tol) #il goal dovrebbe essere da -vic a vic, però il sdr della mappa è posto il modo tale che non ci siamo valori negativi nelle posizioni 
                                                    #(lo zero del sdr map è posto al limite del quadrante negativo)
        goal_y=np.random.uniform(0+tol,2*vic_y-tol)
        theta=np.random.uniform(-np.pi,np.pi) #verificare se theta deve essere generato da 0 a 6.28 o da -3.14 a +3.14
        while any(self.goal_true(goal_x,goal_y,n_obs)): #any restituisce true se c'è almeno un elemento true
            goal_x=np.random.uniform(0,2*vic_x)
            goal_y=np.random.uniform(0,2*vic_y)
        # print("DOPO CORREZIONE: Se c'è un True, non va bene e rigenero il goal (tutti false va bene)",self.goal_true(goal_x,goal_y,n_obs))
        return [goal_x,goal_y,theta]

    def goal_true(self,x,y,n_obs):
        tol=0.2 #tolleranza per considerare l'ingombro totale tra turtlebot e ostacolo
        # g=False
        g=[]
        dist_from_init_pose_to_goal=np.sqrt((abs(self.init_pose[0])-abs(x))**2+(abs(self.init_pose[1])-abs(y))**2) #distanza tra goal e pos attuale turtlebot
        print("distanza dal goal =",dist_from_init_pose_to_goal)
        for i in range(0,n_obs):
            xg=x-self.scene_selected.obs[i].data_pose_obj[0]#traslazione e rotazione ostacolo e goal (sdr al centro dell'ostacolo)
            yg=y-self.scene_selected.obs[i].data_pose_obj[1]
            xg,yg=rot_z(xg,yg,-self.scene_selected.obs[i].data_pose_obj[2])
            if (abs(xg)<=(self.scene_selected.obs[i].l1/2)+tol and abs(yg)<=(self.scene_selected.obs[i].l2/2)+tol):
                g.append(True)
            else:
                if 3<dist_from_init_pose_to_goal<4.5:
                        g.append(False)
                else:    
                    g.append(True)
        return g
    
    def get_quaternion(self, theta):
        q =  quaternion_from_euler (0, 0,theta)
        return q

    def set_init_pose(self):
        initpose = PoseWithCovarianceStamped()
        initpose.header.stamp = rospy.Time.now()
        initpose.header.frame_id = "map"
        # print("x=",self.init_pose[0])
        # print("y=",self.init_pose[1])
        # print("theta=",self.init_pose[2])
        theta=self.init_pose[2]
        initpose.pose.pose.position.x = self.init_pose[0]
        initpose.pose.pose.position.y = self.init_pose[1]
        quaternion = self.get_quaternion(theta)
        initpose.pose.pose.orientation.w = quaternion[0]
        initpose.pose.pose.orientation.x = quaternion[1]
        initpose.pose.pose.orientation.y = quaternion[2]
        initpose.pose.pose.orientation.z = quaternion[3]
        initpose.pose.covariance=[0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
        return initpose
        
    

    def ActionClient_launch(self):
        global data_mmwave_last_value
        global lock
        #define a client for to send goal requests to the move_base server through a SimpleActionClient
        client = actionlib.SimpleActionClient('turtlebot3/move_base',MoveBaseAction)
        client.wait_for_server()
        goal=MoveBaseGoal()
        goal.target_pose.header.frame_id="turtlebot3/map"
        goal.target_pose.header.stamp = rospy.Time.now()
        theta=self.goal_pose[2]
        q=self.get_quaternion(theta)
        goal.target_pose.pose.position =  Point(self.goal_pose[0],self.goal_pose[1],0) # self.goal_pose[0]= x del goal,self.goal_pose[1]= y del goal
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        rospy.loginfo("Sending goal location ...")
        init_time=time.time()
        client.send_goal(goal)
        print("Inizio acquisizione dati run")  
        rate = rospy.Rate(2.0) #Frequenza di lettura dei dati dai sensori (per non far crashare il pacchetto del mmWave è consigliato acquisire a 2~4Hz)
        i=0 #indice usato solo per far variare il numero di frame salvati (immagini e immagini depth)
        format1='rgb8' # salvare le immagini in rgb
        format2='passthrough' # 'passthrough' o '32FC1' per salvare la depthmap
        while (not client.get_state() ==  GoalStatus.SUCCEEDED) and (time.time()-init_time<50) and not rospy.is_shutdown(): #condizione di raggiungimento del goal AND condizione durata run inferiore di 70secondi
            print("Saving sensors data!")
            self.timestamp.append(time.time())
            dir_image=os.path.join(os.getcwd(),'data/run/run'+str(self.ID_run)+'/Image')
            self.data_camera_image.append(dir_image+'/run'+str(self.ID_run)+'_frame'+str(i)+'.png')
            save_image(self.data_camera_image[-1],self.data_camera_image_last_value,format1) #salva le immagini (path,immagine,formato)
            dir_image_depth=os.path.join(os.getcwd(),'data/run/run'+str(self.ID_run)+'/Image_depth')
            self.data_camera_depth.append(dir_image_depth+'/run'+str(self.ID_run)+'_framedepth'+str(i)+'.png')
            save_image(self.data_camera_depth[-1],self.data_camera_depth_last_value,format2) #salva le immagini (path,immagine,formato)
            
            self.data_laser.append(self.data_laser_last_value)
            self.data_imu.append(self.data_imu_last_value)
            self.data_pose.append(self.data_pose_last_value)
            with lock:
                self.data_mmwave.append(data_mmwave_last_value)
                data_mmwave_last_value= []
            i+=1
            rate.sleep()

        self.run_total_time=time.time()-init_time
        print("Fine acquisizione dati run")   
   
        if(client.get_state() ==  GoalStatus.SUCCEEDED):
            print("You have reached the destination")	
            return True
        else:
            print("The robot failed to reach the destination")
            if time.time()-init_time>70:
                print("Run Time > 70 seconds")
            return False

    def moveToGoal(self):

        #Dato che per la navigazione serve caricare la mappa quando si lancia la navigazione (mappa che può cambiare ad ogni run), si lancia nav.launch da 
        #codice per caricare ogni volta una mappa diversa (se si lancia all'inizio il nav (scelta più ovvia e funzionale), si è costretti ad utilizzare una 
        #mappa che deve essere nota già dall'inizio)

        #####Launch file nav.launch 
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        map_selected='map_file:=/home/mike/catkin_ws/src/turtle/script/data/scene/maps/map'+str(self.s_ID) +'.yaml'

        #set pos iniziale (serve solo se si usa amcl, con il vicon non serve perchè abbiamo già la localizzazione perfetta)
        #degli arg settati in cli_args1 viene utilizzato solo map_selected per caricare la mappa. in_x,in_y,in_yaw servono per amcl che è commentato nel launch
        in_x='init_x:='+str(self.init_pose[0])
        in_y='init_y:='+str(self.init_pose[1])
        in_yaw='init_yaw:='+str(self.init_pose[2])
        cli_args1 = ['turtle', 'nav.launch', map_selected,in_x,in_y,in_yaw]
        roslaunch_file1 = roslaunch.rlutil.resolve_launch_arguments(cli_args1)
        roslaunch_args1 = cli_args1[2:]
        launch_files = [(roslaunch_file1[0], roslaunch_args1)]
        parent = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
        parent.start()
        #### end launch
        rospy.sleep(4)

        self.ActionClient_launch()

        parent.shutdown()
   
    

#funzione che conta il numero di ostacoli dai topic
# def count_obj():
#     global long_obs
#     topic_list=rospy.get_published_topics()
#     count=0
#     count_long=0
#     while long_obs!='y' and long_obs!='n':
#         long_obs=input("Utilizzare long Obstacle? (y/n): ")
#     for i in range(0,len(topic_list)):
#         if long_obs=='y':
#             if  topic_list[i][0][0:19]=='/vicon/LongObstacle': #topic_list[i][0][0:15]=='/vicon/Obstacle'
#                 count_long+=1
#         else:
#             if  topic_list[i][0][0:15]=='/vicon/Obstacle':
#                 count+=1
#     if long_obs=='y':
#         return count_long
#     else:
#         return count
    

#funzione che carica tutte le scene presenti nel file data_scene (pickle file) nella lista scene_loaded e salva l'ultimo ID (lastID) dell'ultima scena presente nel file
def read_from_pickle(filename):
    scene_loaded = []
    lastID=-1 #lo faccio partire da -1 in modo tale che, ad ogni lettura di una scena la var lastID contiene in il numero di quella scena e non il successivo (perchè si incrementa dopo la lettura)
    file=filename+"data_scene"
    if os.path.isfile(file):
        with (open(file, "rb")) as openfile:
            while True:
                try:
                    scene_loaded.append(pickle.load(openfile))
                    lastID+=1
                except EOFError:
                    break
        openfile.close()
    return lastID,scene_loaded

def add_to_pickle(filename,item):     
    with open(filename, 'ab') as writefile:
        pickle.dump(item, writefile)
    writefile.close()

def correct_scene(ID,scene_loaded,n_obs,obs_id):
    appo_obj=[]
    go=True
    i=0
    tol=0.08 #tolleranza sulla posizione degli ostacoli (5cm)
    
    if ID>len(scene_loaded)-1 or ID<0:
        print("Scena selezionata non presente!")
        go=False
    else:
        scene=scene_loaded[ID]
        if n_obs!=scene.num_obstacle:
            go=False
        else:
            for j in range (0,n_obs):
                appo_obj.append(obstacle(1,1,j,obs_id)) #creo un ostacolo di dimensioni unitarie perchè mi serve il centro per verificare se la scena scelta è quella attuale
            while i<n_obs and go:
                if scene.obs[i].data_pose_obj[0]-tol<appo_obj[i].data_pose_obj[0]<scene.obs[i].data_pose_obj[0]+tol and scene.obs[i].data_pose_obj[1]-tol<appo_obj[i].data_pose_obj[1]<scene.obs[i].data_pose_obj[1]+tol:
                    go=True
                else:
                    go=False
                    print("La posizione dell'ostacolo "+str(i)+" non coincide con quella presente nella scena selezionata.")    
                    print("Posa ostacolo attuale=",appo_obj[i])
                    print("Posa ostacolo desiderata=",scene.obs[i].data_pose_obj)
                    print("Verificare che gli ostacoli siano nella posizione corretta (entro una tolleranza di 5cm) o di aver selezionato la scena corretta!")    
                i+=1
    return go

def last_run_number(path_run):
    a=os.listdir(path_run) #NB: non sono in ordine
    if a==[]:
        b=-1
    else:
        b=len(a)-1
    return b 

def scene_duplicate(sc,scene_loaded,n_obs,lastID):
    go=False # go indica se la scena letta è uguale ad una già presente nel database delle scene (scene_loaded)
    # go=false-> scena non coincidente con nessuna della lista, 
    # go=true->scena coincidente con una già presente (duplicata)
    i=0
    j=0
    k=0
    tol=0.05
    if lastID!=-1:
        while i<n_obs and not go:
            while j<=lastID and not go:
                if sc.num_obstacle==scene_loaded[j].num_obstacle:
                    while k<=scene_loaded[j].num_obstacle and not go:
                        if sc.obs[i].data_pose_obj[0]-tol<scene_loaded[j].obs[k].data_pose_obj[0]<sc.obs[i].data_pose_obj[0]+tol and sc.obs[i].data_pose_obj[1]-tol<scene_loaded[j].obs[k].data_pose_obj[1]<sc.obs[i].data_pose_obj[1]+tol:
                            go=True #significa che gli ostacoli coincidono con una scena già esistente
                            scene_duplicate=j
                            print("La scena che si sta registrando è già presente nel database (scena n."+str(scene_duplicate)+")")
                            print("Annullamento creazione scena")
                        else:
                            go=False
                        k+=1
                j+=1
            i+=1
    else:
        go=False
    return go

def scene_builder(filename,lastID,scene_loaded,obs_id):
    
    #1)costruzione scena e aggiornamento scene_loaded e pickle file:
    # n_o=count_obj()
    print("Numero di ostacoli presenti nella scena: "+str(len(obs_id)))
    y=''
    if len(obs_id)>=1:
        while y!='y' and y!='Y':
            y=input("Posizionare gli ostacoli nella scena e successivamente digitare y: ")
        s=scene(lastID+1,obs_id)            
        if not scene_duplicate(s,scene_loaded,len(obs_id),lastID): #return false se la scena NON è duplicata
            scene_loaded.append(s)
            rospy.sleep(0.1)
            add_to_pickle(filename+"data_scene",s)
            # create_map(filename,s,n_o,lastID+1)  #funzione per creare la mappa automaticamente dalle coordinate del Vicon
            print("La scena appena registrata è la numero: "+str(lastID+1))
            lastID+=1
    else:
        print("Non ci sono ostacoli nella scena, aggiungere almeno 1 ostacolo per la creazione di una scena")
    # exit()
    return lastID,scene_loaded

def obstacle_to_plot(s):
    plt.figure()
    print("n_obs =",s.num_obstacle)
    for i in range(0,s.num_obstacle):
        b=s.obs[i].l1
        h=s.obs[i].l2
        py=np.array((-h/2, -h/2,h/2,h/2,-h/2))
        px=np.array((-b/2, b/2,b/2,-b/2,-b/2))
        for j in range(0,len(px)):
            px[j],py[j]=rot_z(px[j],py[j],s.obs[i].data_pose_obj[2])
            px[j]+=s.obs[i].data_pose_obj[0]#+vic_x
            py[j]+=s.obs[i].data_pose_obj[1]#+vic_y
        plt.plot(px,py,'k') 
    border=[[0,0],[2*vic_x,0],[2*vic_x,2*vic_y],[0,2*vic_y],[0,0]]
    borderx, bordery = zip(*border) #create lists of x and y values
    plt.plot(borderx,bordery,'k')
    

# def create_map(filename,s,n_obs,num_map):  # creazione automatica della mappa leggendo i dati del vicon (deve essere scalata, quindi per ora la lascio commentata)
#     #mappa fatta dalla slam ha dimensioni (384, 384, 3)
#     fig = plt.figure()
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     obstacle_to_plot(s,n_obs)
#     tol=0.05
#     extern=[[-vic_x-tol,vic_y-tol],[vic_x+tol,-vic_y-tol],[vic_x+tol,vic_y+tol],[-vic_x-tol,vic_y+tol],[-vic_x-tol,-vic_y-tol]]
#     r2=patches.Polygon(extern,color='#cdcdcd')
#     ax.add_patch(r2)
#     border=[[-vic_x,vic_y],[vic_x,-vic_y],[vic_x,vic_y],[-vic_x,vic_y],[-vic_x,-vic_y]]
#     borderx, bordery = zip(*border) #create lists of x and y values
#     plt.plot(borderx,bordery,'k')
#     r1=patches.Polygon(border,color='white')
#     ax.add_patch(r1)
#     fig.savefig('map.png', bbox_inches='tight',pad_inches = 0)
#     im = Image.open('map.png')
#     im = im.convert('RGB')
# #     map_name=filename+'maps/map'+str(num_map) #senza estensione .pgm o .yaml
#     map_name='/home/mike/Desktop/map'
#     im.save(map_name+'.pgm')
#     exit()
#     os.remove("map.png") 
#     yaml_name=map_name+'.yaml'
#     map_name='./map'+str(num_map)
#     res= 0.05 #2*max_vic/np.shape(im)[0]
#     create_yaml(yaml_name,map_name,res)
       
# def create_yaml(yaml_name,map_name,res):
#     mapName='map'
#     yaml = open(yaml_name, "w")
#     yaml.write("image: " +map_name+ ".pgm\n")
#     # yaml.write("resolution: 0.050000\n")
#     yaml.write("resolution: "+str(res)+"\n")
#     yaml.write("origin: [" + str(-1) + "," +  str(-1) + ", 0.000000]\n")
#     yaml.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196")
#     yaml.close()

def plot_scene(lastID):
    # 2)plot delle scene
    for i in range(0,lastID+1):
        map_path="data/scene/maps/map"+str(i)+".pgm"
        map_path=os.path.join(os.getcwd(), map_path)
        im=cv2.imread(map_path)
        # obstacle_to_plot()
        plt.figure()
        plt.title('Scene '+str(i))
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.axis('equal')
        plt.imshow(im)
    plt.show()

def single_run(scene_loaded,r,lastID,path_run,n_obs,obs_id):
    #3)RUN
    print("Ci sono "+str(lastID+1)+" scene (numerate da 0 a "+str(lastID)+")") #lastID+1 perchè la numerazione delle scene parte da 0
    ID_sel_scene=int(input('Quale scena utilizzare?:'))
    while not correct_scene(ID_sel_scene,scene_loaded,n_obs,obs_id):
        ID_sel_scene=int(input('Quale scena utilizzare?:'))
    i=last_run_number(path_run)+1 #numero della nuova run (leggendo il numero dell'ultima run già presente nella directory run)
    os.mkdir(path_run+"run"+str(i))
    os.mkdir(path_run+"run"+str(i)+'/Image')
    os.mkdir(path_run+"run"+str(i)+'/Image_depth')
    r=run(i,ID_sel_scene,scene_loaded[ID_sel_scene],n_obs,obs_id)
    add_to_pickle(path_run+"run"+str(i)+"/data_run"+str(i),r)

def multi_run(scene_loaded,r,lastID,path_run,n_obs,obs_id):
    #4) MULTIPLE RUN
    print("Ci sono "+str(lastID+1)+" scene (numerate da 0 a "+str(lastID)+")")
    ID_sel_scene=int(input('Quale scena utilizzare?:'))
    while not correct_scene(ID_sel_scene,scene_loaded,n_obs,obs_id):
        ID_sel_scene=int(input('Quale scena utilizzare?:'))
    n_run=int(input('Quante run fare?:'))
    i=last_run_number(path_run)+1 #numero della nuova run (last_run_number ritorna il numero dell'ultima run già presente nella directory run)
    for k in range (i,n_run+i):
        os.mkdir(path_run+"run"+str(k))
        os.mkdir(path_run+"run"+str(k)+'/Image')
        os.mkdir(path_run+"run"+str(k)+'/Image_depth')
        r=run(k,ID_sel_scene,scene_loaded[ID_sel_scene],n_obs,obs_id)
        add_to_pickle(path_run+"run"+str(k)+"/data_run"+str(k),r)
        rospy.sleep(1)

def get_key(val):
    global obs_topic
    for key, value in obs_topic.items():
         if val == value:
             return key
    return "Obstacle doesn't exist (No Topic)!"

def menu(path_scene,path_run,topic_to_see):
    global obs_topic
    r=[]    #lista che conterrà le run
    obs_id=[]
    for i in range(0,len(topic_to_see)):
        obs_id.append(get_key(topic_to_see[i]))
    n_obs= len(obs_id) #count_obj(topic_to_see)
    lastID,scene_loaded=read_from_pickle(path_scene) # lastID contiene l'ultimo ID della scena presente nel file, 
                                                     # scene_loaded contiene tutte le scene caricate dal file
    flag=True
    while flag:
        print("Menù:")
        print("1)Crea una scena")
        print("2)Plotta le scene")
        print("3)Esegui una singola Run")
        print("4)Esegui Run multiple (stessa scena)")
        print("5)Exit")
        c=int(input('Scelta(numero):'))
        if c!=5:
            while not c in [1,2,3,4,5]:
                print("Scelta errata, inserisci un numero accettabile")
                c=int(input('Scelta(numero):'))

            while (c!=1) and lastID==-1:
                print("Nessuna scena registrata,seleziono la creazione di una scena!")
                c=1
            
            if c==1:
                [lastID,scene_loaded]=scene_builder(path_scene,lastID,scene_loaded,obs_id) #faccio ritornare la lista delle scene caricate aggiornata con la nuova scena creata
            elif c==2:
                plot_scene(lastID)
                # for i in range(0,len(scene_loaded)):
                #     obstacle_to_plot(scene_loaded[i]) 
                # plt.show()
            elif c==3:
                single_run(scene_loaded,r,lastID,path_run,n_obs,obs_id)
            
            elif c==4:
                multi_run(scene_loaded,r,lastID,path_run,n_obs,obs_id)

        else:
            flag=False
    print("......The End......")

def main():
    # topic_to_see=['/vicon/LongObstacle01/LongObstacle01','/vicon/LongObstacle23/LongObstacle23'] #scena 0
    topic_to_see=['/vicon/Obstacle0/Obstacle0','/vicon/Obstacle1/Obstacle1','/vicon/Obstacle2/Obstacle2','/vicon/Obstacle3/Obstacle3'] #scena 1
        #Se si da lo stesso nome ai topic con numerazione crescente, si può usare la funzione count_obj che conta gli ostacoli dai topic e prende direttamente il nome.
        #se si utilizzano contemporaneamente nomi diversi (tipo ObstacleX e LongObstacleX) come in questo caso, non va bene e si usano i topic scritti a mano (topic_to_see)
    rospy.init_node('dataset')
    path_run=os.path.join(os.path.dirname(__file__), 'data/run/')
    path_scene=os.path.join(os.path.dirname(__file__), 'data/scene/') #nome/percorso del file che contiene le scene salvate
    menu(path_scene,path_run,topic_to_see)
    

if __name__ == '__main__':
    main()