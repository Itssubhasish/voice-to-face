voice_mel = np.load('vox_celeb/id10045/xcGtfOT57-M/00001.npy')
#plt.figure(figsize = (16,8))
#fig, ax = plt.subplots(1,2)
plt.imshow(voice_mel[1:100,:].T)
plt.colorbar()
plt.show()
#plt.imshow(voice_mel[0:8,0:8])
#print(voice_mel[0:8,0:8])




#plt.figure(figsize = (24,10))
for i in range(1,64):
  plt.plot(voice_mel[:,i])



