import PySimpleGUI as sg
import face_Recognition as fr
import face_Detection as fd
import numpy as np
import cv2
import os
dir_path=os.path.dirname(os.path.realpath(__file__))
data=np.load("./workers/olivetti_faces.npy")
target=np.load("./workers/olivetti_faces_target.npy")
n_workers=40
def load_win1(model):
    layout=[[sg.Text("Benvinguts a Company Vision")],
    [sg.Text("Per poder accedir a la següent sala, has de tenir autorització")],
    [sg.Text("Per comprovar si tens l'autorització permi el botó:")],
    [sg.Button("Identifica'm")],[sg.Button("Afegeix Treballador")]]
    window=sg.Window("COMPANY VISION",layout,size=(300,200))
    id=False
    while True:
        event,values=window.read()
        if event=="Identifica'm":
            id=True
            break
        if event=="Afegeix Treballador":
            id=False
            break
    if id==True:
        window.close()
        load_win2(model)
    else:
        window.close()
        load_win4(model,data,target,n_workers)
def load_win2(model):
    i=0
    cap = cv2.VideoCapture(0)
    _, im = cap.read()#depen de la llum que tens, pot ser que no et detecti la cara
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_detect=fd.face_detect(img)
    if len(im_detect)!=0:
        path=os.path.join(dir_path,"persona{}.png".format(i))
        cv2.imwrite(path, im_detect[0])
        
        layout=[[sg.Text("A continuació verificarem la seva identitat")],
        [sg.Text("Imatge a reconeixer:")], [sg.Image(path)], [sg.Button("OK"),
        [sg.Button("TORNA")]]
        ]
        window=sg.Window("Recognizing...",layout,size=(300,200))
        
        event,values=window.read()
        if event=="OK":
            im1=[]
            im1.append(im_detect[0])
            im1=np.array(im1)/255
            im1.resize(len(im1), 64, 64, 1)
            window.close()
            load_win3(im1,model)
            
        if event=="TORNA":
            window.close()
            load_win1(model)
        
    else:
        layout=[[sg.Text("No s'ha detectat cap cara, torna a intentar-ho!")],
        [sg.Button("OK")]
        ]
        window=sg.Window("ERROR...",layout,size=(300,200))
        event,values=window.read()
        if event=="OK":
            window.close()
            load_win1(model)
        
    
    
def load_win3(img,model):
    th=0.9
    res=model.predict(img)
    prob=np.max(res)
    iprob=np.argmax(res)
    if prob>th and iprob!=n_workers:#Permet entrar
        layout=[[sg.Text("Resultat del Reconeixament:")],
        [sg.Text("TENS LA ENTRADA AUTORITZADA!")],
        [sg.Button("D'ACORD")]]
        window=sg.Window("RESULTS",layout)
        op=0
        
        event,values=window.read()
        if event=="D'ACORD":
            window.close()
            load_win1(model)

    else:#Entrada denegada
        layout=[[sg.Text("Resultat del Reconeixament:")],
        [sg.Text("TENS LA ENTRADA DENEGADA!")],
        [sg.Button("D'ACORD")],[sg.Button("TORNA A INTENTAR")]]
        window=sg.Window("RESULTS",layout)
        
       
        event,values=window.read()
        if event=="D'ACORD":
            window.close()
            load_win1(model)
        if event=="TORNA A INTENTAR":
            window.close()
            load_win1(model)
        
        
    
def load_win4(model,data,target,n_workers):#Add Worker
    layout=[[sg.Text("Esteu afegint un nou treballador...")],
    [sg.Text("Per poder afegir un treballador a la Base de Dades cal insertar imatges seves")],
    [sg.Text("A continuació pots indicar la carpeta on es troben les imatges:")],
    [sg.Text("Tria una carpeta: "), sg.FolderBrowse(key="carpeta")],
    [sg.Button("INSERTA")],[sg.Button("TORNA")]]
    window=sg.Window("ADD WORKER",layout,size=(600,300))
    
    event,values=window.read()
    if event=="INSERTA":
        id=True #Aqui insertar les imatges del treballador
        n_workers+=1
        ims=[]
        files=[]
        label=max(target)+1
        for file in os.listdir(values['carpeta']):
            if '.jpg' or '.png' in file:
               files.append(file)
        for i in files:
            p=os.path.join(values['carpeta'],i)
            im=cv2.imread(p)
            im_g=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_r=cv2.resize(im_g, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            im_n=im_r/255
            im_n.resize(1,64,64)
            data=np.append(data,im_n,axis=0)
            target=np.append(target,label)

        x_train,y_train,x_test,y_test=fr.prepare_workers(data,target,8,2)
        model,history=fr.create_model(x_train,y_train,x_test,y_test,n_workers)
        fr.show_results(history)

        window.close()
        load_win1(model)
    if event=="TORNA":
        window.close()
        load_win1(model)

    
if __name__ == '__main__':
    """
    Preparar el model amb els treballadors de la empresa i mostrar resultats
    """
    x_train,y_train,x_test,y_test=fr.prepare_workers(data,target,8,2)
    model,history=fr.create_model(x_train,y_train,x_test,y_test,n_workers)
    #fr.show_results(history)
    load_win1(model)
    
    




    
