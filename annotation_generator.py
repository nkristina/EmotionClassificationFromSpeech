import os
import re
import pandas as pd
import numpy as np

ravdess_path = r"C:\Users\psiml\Desktop\PSIML_projekat\dataset\Ravdess\audio_speech_actors_01-24"
ravdess_directory_list = os.listdir(ravdess_path)

file_emotion = [[],[],[]]
file_path = [[],[],[]]
file_gender = [[],[],[]]
file_intensity = [[],[],[]]
file_semantics = [[],[],[]]

for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(ravdess_path + '\\' + dir)
    
    id = re.findall(r'\d+', dir)[0]

    gender = ''
    if int(id) % 2 == 0:
        gender = 'female'
    else:
        gender = 'male'

    if int(id) <=18:
        set = 0
    elif int(id) <=22:
        set = 1
    elif int(id) <=24:
        set = 2
    else:
        continue

    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        
        file_emotion[set].append(int(part[2]))

        if int(part[2]) in [1,2,3,8]:
            file_semantics[set].append(0)
            #file_semantics[set].append('positive')
        else:
            file_semantics[set].append(1)
            #file_semantics[set].append('negative')

        file_path[set].append(os.path.join(ravdess_path,dir,file))
        file_intensity[set].append(int(part[3]))
        file_gender[set].append(gender)


# Train Dataframes

emotion_train_df = pd.DataFrame(file_emotion[0], columns=['Emotions'])
gender_train_df = pd.DataFrame(file_gender[0], columns=['Gender'])
intensity_train_df = pd.DataFrame(file_intensity[0], columns=['Intensity'])
semantics_train_df = pd.DataFrame(file_semantics[0], columns=['Semantics'])
path_train_df = pd.DataFrame(file_path[0], columns=['Path'])

Ravdess_train_df = pd.concat([emotion_train_df, semantics_train_df, intensity_train_df, gender_train_df, path_train_df], axis=1)

Ravdess_train_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_train_df.Intensity.replace({1: 'normal', 2:'strong'}, inplace=True)

Ravdess_train_df.to_csv(r"C:\Users\psiml\Desktop\PSIML_projekat\Ravdess_train.csv",index=False)

# Validation Dataframes

emotion_val_df = pd.DataFrame(file_emotion[1], columns=['Emotions'])
gender_val_df = pd.DataFrame(file_gender[1], columns=['Gender'])
intensity_val_df = pd.DataFrame(file_intensity[1], columns=['Intensity'])
semantics_val_df = pd.DataFrame(file_semantics[1], columns=['Semantics'])
path_val_df = pd.DataFrame(file_path[1], columns=['Path'])

Ravdess_val_df = pd.concat([emotion_val_df, semantics_val_df, intensity_val_df, gender_val_df, path_val_df], axis=1)

Ravdess_val_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_val_df.Intensity.replace({1: 'normal', 2:'strong'}, inplace=True)

Ravdess_val_df.to_csv(r"C:\Users\psiml\Desktop\PSIML_projekat\Ravdess_validation.csv",index=False)

# Test Dataframes

emotion_test_df = pd.DataFrame(file_emotion[2], columns=['Emotions'])
gender_test_df = pd.DataFrame(file_gender[2], columns=['Gender'])
intensity_test_df = pd.DataFrame(file_intensity[2], columns=['Intensity'])
semantics_test_df = pd.DataFrame(file_semantics[2], columns=['Semantics'])
path_test_df = pd.DataFrame(file_path[2], columns=['Path'])

Ravdess_test_df = pd.concat([emotion_test_df, semantics_test_df, intensity_test_df, gender_test_df, path_test_df], axis=1)

Ravdess_test_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_test_df.Intensity.replace({1: 'normal', 2:'strong'}, inplace=True)

Ravdess_test_df.to_csv(r"C:\Users\psiml\Desktop\PSIML_projekat\Ravdess_test.csv",index=False)
