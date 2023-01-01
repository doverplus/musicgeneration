#https://blog.csdn.net/weixin_33841498/article/details/113512845
#https://mido.readthedocs.io/en/latest/midi_files.html#iterating-over-messages


from mido import MidiFile
import os

path='train_midifiles' # address of midifiles
files= os.listdir(path)
txts = []

for file in files: 
    if file.endswith('mid') or file.endswith('midi'):           
        position = path+'\\'+ file 
        print (position)
        outputaddress=position+'output' # data will be print in '.midoutput' files.
        output=open(outputaddress, mode='w', encoding='UTF-8')
        midifile = MidiFile(position)
        tracks = midifile.tracks
        for i, track in enumerate(midifile.tracks):
            output.write('Track {}: {}'.format(i, track.name)+'\n') 
        for i, track in enumerate(midifile.tracks):
            output.write('\n')
            output.write('Track {}: {}'.format(i, track.name)+'\n')            
            for msg in track:
                output.write(str(msg)+'\n')                
        output.close