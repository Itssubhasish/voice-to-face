


for name in VGG_name:
    if os.path.exists(os.path.join(face_data, name)):
        print("Found -->", os.path.join(face_data, name))
    else:
        print("Did not find -->", os.path.join(face_data, name))




for name in Celeb_id:
    if os.path.exists(os.path.join(voice_data, name)):
        print("Found -->", os.path.join(voice_data, name))
    else:
        print("Did not find -->", os.path.join(voice_data, name))


n = 7
fig, axes = plt.subplots(1, n, figsize=(20,10))

for i in range(n):
    random_face = random.choice(face_names)
    random_image = random.choice(os.listdir(os.path.join(face_data, random_face)))
    random_image_file = os.path.join(face_data, random_face, random_image)
    image = plt.imread(random_image_file)
    axes[i].imshow(image)
    axes[i].set_title("Celeb: " + random_face.replace('_', ' '))
    axes[i].axis('off')

plt.show()




'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles        
def main():
    c = 0
    dirName = 'vox_celeb';
    
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    
    # Print the files
    for elem in listOfFiles:
        print(elem)
    print ("****************")
    
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        
        
    # Print the files    
    for elem in listOfFiles:
        c+= 1
        #print(elem)    
        
    print(c)      
        
        
if __name__ == '__main__':
    main()
