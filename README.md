# multimodal_dataset

Insieme di tool volto alla creazione di un dataset multimodale utilizzando il robot mobile “turtlebot3 waffle pi” il cui comparto sensoristico è stato ampliato con l’aggiunta di una depth camera e un sensore mmWave.  
Inoltre, un grande valore aggiunto è dato dall'utilizzo del sistema di motion capture Vicon che fornisce la posa del robot e degli oggetti presenti all'interno delle mappe con una precisione pressoché assoluta.

ESECUZIONE:
IN SSH (su turtlebot): 
- roslaunch turtlebot3_bringup turtlebot3_robot.launch

SU PC:
- roslaunch turtle start_turtle.launch

1) Creazione manuale della mappa:
  - roslaunch turtle gmapping_custom.launch 
  - roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch (movimentazione robot da tastiera)  
  Dopo l'esplorazione della mappa (non arrestare gmapping!):
  - roslaunch turtle save_map.launch  
  La mappa sarà salvata in turtle/map_temp e potrà essere successivamente spostata in turtle/script/data/scene/maps e rinominata opportunamente col nome    della scena corrispondente (NB: ciò vale anche per il file .yaml)
  
2) Creazione dataset:
  - rosrun turtle dataset.py
 

Tools aggiuntivi: 
1) Conversione dati delle run da pickle a json: 
    - rosrun turtle dataset_to_json.py  
      (file json salvati nella cartella turtle/script/data/json_conversion)
2) Creazione di una depthmap utilizzando immagine e pontcloud:
    - rosrun turtle create_depthmap_from_mmwave_image.py
3) Plot tutti i punti di una run con calcolo della chamfer distance:
    - rosrun turtle complete_mmwave_map_run.py
