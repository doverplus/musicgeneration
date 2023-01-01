import midimatrix
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import random

EPOCHS=2
FEATURE_NUMBER=60
TRAIN_WINDOW=128
CUT_POINT_OF_PREDICT_SOURCE=0.4

HIDDEN_SIZE=40
TOTAL_NOTE_NUMBER=128
PREDICT_LENGTH=256


#get all music data from dir
def getMidimatrix():
    number_of_midi=midimatrix.getNumberOfMidi(midimatrix.TRAIN_MIDI_PATH)
    print(str(number_of_midi)+' midi files are used as training data.')
    datasetlist=midimatrix.getTrainMatrix()
    dataset=[] 
    for i in range(number_of_midi):
        dataset=dataset+datasetlist[i]
    return dataset


#throw away some note for speed training or special performance
def removeNote(dataset):
    new_dataset=[]
    cut_number=int((TOTAL_NOTE_NUMBER-FEATURE_NUMBER)/2)
    for note in dataset:
        new_dataset.append(note[cut_number:-cut_number])
    return new_dataset


def intMatrixToFloat(int_matrix):
    a=[]
    for i in range(len(int_matrix)):   
        a.append(list(map(float,int_matrix[i])))
    return a    
           
#each element of matrix represent the possiblity that there is a note on.             
def chooseNote(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j]<0.01:
                matrix[i][j]=0
            if matrix[i][j]>0.99:
                matrix[i][j]=1
            if 0.01<matrix[i][j]<0.99:
                if random.random()<matrix[i][j]:
                    matrix[i][j]=1
                else:
                    matrix[i][j]=0                                
    return matrix 

#batch training sequences
def createTrainSeqs(input_data, tw):
    train_seqs = [] 
    L = len(input_data)
    for i in range(L-tw-1):
        train_seq = input_data[i:i+tw]    
        train_seqs.append(train_seq)   
    return train_seqs

#batch training labels
def createTrainLabels(input_data, tw,num_note):
    train_labels=[]
    L = len(input_data)
    for i in range(L-tw-1):
        train_label = input_data[i+tw+1][num_note]
        train_labels.append(train_label)
    return train_labels    


#LSTM model
class LSTM(nn.Module):
    def __init__(self,input_size=FEATURE_NUMBER, hidden_size=HIDDEN_SIZE, output_size=1):
        super().__init__()
        self.hidden_size=hidden_size
        self.lstm=nn.LSTM(input_size,hidden_size)
        self.linear=nn.Linear(hidden_size,output_size)
        self.hidden_cell = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))#

    def forward(self, input):
        output,self.hidden_cell=self.lstm(input,self.hidden_cell)
        predictions = self.linear(output.view(len(input), -1))
        return predictions[-1]



#train one feature model
def trainFeature(train_seqs,train_labels):
    model = LSTM()
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model.to(device)

    for i in range(EPOCHS):
        for j in range(len(train_seqs)):
            seq=train_seqs[j].view(-1,1,FEATURE_NUMBER)
            labels=train_labels[j]
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_size).to(device),
                            torch.zeros(1, 1, model.hidden_size).to(device))
            y_pred = model(seq)
            single_loss = loss(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        if i%1 == 0:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
           
    return model 


#predict a new note according to previous trained model, repeat this process FEATURE_NUMBER times to get a new note row.
def predictNoteRow(input,modellist):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    seq = input.reshape(-1,1,FEATURE_NUMBER).to(device)
    predictions=[]
    for i in range(FEATURE_NUMBER):
        with torch.no_grad():
            modellist[i].hidden_cell = (torch.zeros(1, 1, model.hidden_size).to(device),
                            torch.zeros(1, 1, model.hidden_size).to(device))
            predictions.append(modellist[i](seq).item())        
    return predictions

#put back the removed number and choose the onnote and offnote.
def getPredictMatrix(realpred,cut_number):
    predict_matrix=[]
    for row in realpred:
        m=[]
        zeros=[0]*cut_number
        m=m+zeros
        m=m+list(row)
        m=m+zeros    
        predict_matrix.append(m)
    actual_pred_matrix=chooseNote(predict_matrix)    
    return actual_pred_matrix


#get pred input matrix(shape=TRAIN_WINDOWS*FEATURE_NUMBER)
def getpredInput(cutnumber,index):
    pred_input=midimatrix.getPredictMatrix(index)
    pred_input=removeNote(pred_input)
    cutnote=int(cutnumber*len(pred_input))
    pred_input=pred_input[cutnote:cutnote+TRAIN_WINDOW]
    scaler=StandardScaler()
    sca_pred_input=scaler.fit_transform(pred_input)
    return sca_pred_input


#add new noterow to the end of pred_matrix 
def getNewPredMatrix(pred_matrix,new_noterow):
    pred_matrix=pred_matrix.reshape(-1,FEATURE_NUMBER)
    pred_matrix = pred_matrix.to('cpu')
    pred_matrix=pred_matrix.numpy()
    pred_matrix=pred_matrix.tolist()
    pred_matrix.append(new_noterow)
    new_pred_matrix=pred_matrix[1:] 
    pred_matrix = torch.FloatTensor(new_pred_matrix).view(-1,1,FEATURE_NUMBER)
    return pred_matrix

if __name__ == '__main__':
    #get dataset
    dataset=getMidimatrix()
    dataset=removeNote(dataset)
    removed_number=int((TOTAL_NOTE_NUMBER-FEATURE_NUMBER)/2)
    print('Number of data(fragment): '+str(len(dataset)))
    print('Number of note(feature): '+str(len(dataset[0])))

    #get standardized data for training
    dataset=intMatrixToFloat(dataset)
    t_scaler=StandardScaler()
    sca_dataset=t_scaler.fit_transform(dataset)

    #batch our data to input sequences
    train_sequences=createTrainSeqs(sca_dataset,TRAIN_WINDOW)
    n_train_labels=[]
    for i in range(FEATURE_NUMBER):
        n_train_labels.append(createTrainLabels(sca_dataset,TRAIN_WINDOW,i))
    
    #gpu is faster than cpu for training
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    
    #train all the model(number=feature_number)
    modellist=[]
    train_sequences = torch.FloatTensor(train_sequences).view(-1,TRAIN_WINDOW,FEATURE_NUMBER)
    train_sequences=train_sequences.to(device)
    for i in range(FEATURE_NUMBER):
        train_labels =torch.FloatTensor(n_train_labels[i]).view(-1,1)
        train_labels=train_labels.to(device)
        print('start training note '+str(removed_number+i))
        model=trainFeature(train_sequences,train_labels)
        modellist.append(model)

    #each for loop predict one midi from source music.
    for j in range(midimatrix.getNumberOfMidi(midimatrix.PREDICT_MIDI_PATH)):
        print('start predicting the '+str(j)+'music')
        #predict a new noterow, then add this new row to the end of pred_matrix, repeat
        n_prediction=[]
        pred_matrix=getpredInput(CUT_POINT_OF_PREDICT_SOURCE,j)
        pred_matrix = torch.FloatTensor(pred_matrix).view(-1,1,FEATURE_NUMBER)
        pred_matrix=pred_matrix.to(device)
        for i in range(PREDICT_LENGTH):#budui
            print('predicting note '+str(i))
            new_noterow=predictNoteRow(pred_matrix,modellist)
            pred_matrix=getNewPredMatrix(pred_matrix,new_noterow)
            pred_matrix.to(device)
            n_prediction.append(new_noterow)
            
        #inverse predict data, put back the removed number and choose the onnote and offnote.
        if PREDICT_LENGTH>TRAIN_WINDOW:
            realpred=t_scaler.inverse_transform(n_prediction)
            realpred=getPredictMatrix(realpred,removed_number)
        else:        
            pred_matrix=pred_matrix.reshape(-1,FEATURE_NUMBER)
            realpred=t_scaler.inverse_transform(pred_matrix.cpu())
            realpred=getPredictMatrix(realpred,removed_number)

        #output predict matrix to midifile    
        pred_midi_name='predsource-'+midimatrix.getPredictSource(j)+str(CUT_POINT_OF_PREDICT_SOURCE)\
            +'_trainnum'+str(midimatrix.getNumberOfMidi(midimatrix.TRAIN_MIDI_PATH))\
            +'_epoch'+str(EPOCHS)+'_memorylen'+str(TRAIN_WINDOW)\
            +'_predlen'+str(PREDICT_LENGTH)+'_feature'+str(FEATURE_NUMBER)+'.mid'
        midimatrix.matrixToMidi(realpred,pred_midi_name)    
   