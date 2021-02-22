x, fs = librosa.load('Eoin_Macken.wav')
librosa.display.waveplot(x, sr=fs)



IPython.display.Audio(x, rate=fs)





mfccs = librosa.feature.mfcc(x, sr=fs)
print(mfccs.shape)
librosa.display.specshow(mfccs, sr=fs, x_axis='time')




##Let's scale the MFCCs such that each coefficient dimension has zero mean and unit variance:
# Feature scaling


mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))





librosa.display.specshow(mfccs, sr=fs, x_axis='time')      ### Plot standarized mfcc