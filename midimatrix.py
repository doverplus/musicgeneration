
# most used number of ticks per beat:480, 720, 960, 1024, 1200, 1440, 1920, 2880
# tick division could be: 30, 60, 64, 120, 240
# Following code in method 'tensortomidi' will affect the rhythm of the output music
# MetaMessage('set_tempo', tempo=2000000, time=0) 
# MetaMessage('time_signature', numerator=4, denominator=8, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0)

from mido import MetaMessage, Message, MidiFile, MidiTrack
import os


TICK_DIVISION=60  
NUMBER_OF_NOTE=128
DEFAULT_VELOCITY=60
TRAIN_MIDI_PATH='train_midifiles' # address of midifiles
PREDICT_MIDI_PATH='predict_midi'

          
def midifileToMatrix(midifile,tickdivision):
    submatrix=[[] for i in range(len(midifile.tracks))]       
    noteon=[]    
    for m in range(0,len(midifile.tracks)):
        numberofdivision=0
        ticks=0
        for msg in midifile.tracks[m]: 
            if isinstance(msg, Message) & hasattr(msg, 'velocity'): # if is a message with time and note                 
                # 1: noteon velocity=x time=0 : add note
                # 2: noteon velocity=0 time=0 : remove note
                # 3: noteon velocity=x time=y : read noteon, pass time, add note
                # 4: noteon velocity=0 time=y : read noteon, pass time, remove note
                # 5: noteoff time=0 : remove note
                # 6: noteoff time=y : read noteon, passtime, remove note
                if msg.velocity !=0 and msg.time == 0 and msg.type=='note_on': #1                 
                    noteon.append(msg.note)                    
                elif (msg.velocity ==0 or msg.type=='note_off') and  msg.time == 0: #2 #5                        
                    noteon.remove(msg.note)
                elif msg.time != 0  : #when time changed, print all note during the time period in one row. 
                    notelist=[0]*NUMBER_OF_NOTE
                    ticks+=msg.time
                    for i in range(0,len(noteon)):                         
                        notelist[noteon[i]]=1                        
                    for i in range(0,int((ticks-numberofdivision*tickdivision)/tickdivision)): 
                        submatrix[m].append(notelist)
                        numberofdivision+=1
                    if msg.velocity !=0 and msg.type=='note_on': #3
                        noteon.append(msg.note)                    
                    else:                                        #4 #6         
                        noteon.remove(msg.note)
    lenoftracks=[]
    for i in range(0,len(submatrix)):
        lenoftracks.append(len(submatrix[i]))    
    matrix=[[0 for j in range(NUMBER_OF_NOTE) ] for i in range(max(lenoftracks))]
    for i in range(0,len(matrix)):
        for m in range(len(submatrix)):            
            if(i<len(submatrix[m])):
                for j in range(0,127):
                    if submatrix[m][i][j]==1:
                        matrix[i][j]=1                                                            
    return matrix                        

def matrixToMidi(matrix,filename): 
    path=os.path.split(os.path.realpath(__file__))[0]+'/output_midiflies' # address of newmidifiles
    output= path+'\\'+filename
    mid = MidiFile()
    track0 = MidiTrack()
    track = MidiTrack()
    mid.tracks.append(track0)
    mid.tracks.append(track)
    track0.append(MetaMessage('time_signature', numerator=4, denominator=8, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    track0.append(MetaMessage('set_tempo', tempo=2000000, time=0)) #add the metamessage
    for i in range(len(matrix)):
        for j in range(NUMBER_OF_NOTE):
            #1: i=0 note[i]=1: noteon
            #2: i!=0 note[i-1]=0 and note[i]=1: noteon
            #3: i!=len note[i-1]=1 and note[i]=0: noteoff
            #4: i=len note[i]=1: note off
            if i==0 and matrix[i][j]==1: #1
                track.append(Message('note_on', note=j, velocity=DEFAULT_VELOCITY, time=0))
            if i!=0 and matrix[i-1][j]==0 and matrix[i][j]==1: #2
                track.append(Message('note_on', note=j, velocity=DEFAULT_VELOCITY, time=0)) 
            if i!=len(matrix)-1 and matrix[i-1][j]==1 and matrix[i][j]==0:  #3
                track.append(Message('note_off', note=j, velocity=0, time=0))
            if i==len(matrix)-1 and matrix[i][j]==1: #4
                track.append(Message('note_off', note=j, velocity=0, time=0))
        track.append(Message('note_on', note=0, velocity=0, time=TICK_DIVISION))               
    mid.save(output)
    print(filename+' created.')



def getMidimatrixFromDir(path):
    matrix_list=[]
    files= os.listdir(path)
    for file in files: 
        if file.endswith('mid') or file.endswith('midi'):          
            position = path+'\\'+ file 
            midifile = MidiFile(position)
            m=midifileToMatrix(midifile,TICK_DIVISION) 
            matrix_list.append(m) 
            print('Load midi '+str(file)+'.')        
    return  matrix_list    


def getTrainMatrix():
    train_matrix_list=getMidimatrixFromDir(TRAIN_MIDI_PATH)
    return train_matrix_list      


def getNumberOfMidi(path):
    number_midi=0
    files= os.listdir(path)
    for file in files: 
        if file.endswith('mid') or file.endswith('midi'):
            number_midi=number_midi+1
    return number_midi 


def getPredictMatrix(index): 
    predict_matrix_list=getMidimatrixFromDir(PREDICT_MIDI_PATH)
    return predict_matrix_list[index]     


def getPredictSource(index):
    files= os.listdir(PREDICT_MIDI_PATH)
    name_list=[]
    for file in files: 
        if file.endswith('mid'):
            name_list.append(str(file)[:-4])
        if file.endswith('midi'):
            name_list.append(str(file)[:-5])
    return name_list[index]


