import os
import pandas as pd

#Create the csv file with the paths of the deepfake examples

#ds_path: Path of the original test csv
#samples: int number corresponding to the number of frames to sample for each video
#return: Path of the created csv

def getFFPath(ds_path,samples=10):
    #Create the name of the new csv
    save_path = ds_path.split('/')
    save_path[-1] = save_path[-1].split('.')[0] + '_final.csv'
    save_path = '/'.join(save_path)

    #Return the path if the csv already exists
    if(os.path.isfile(save_path)):
        return save_path

    #Read the original test csv
    df = pd.read_csv(ds_path, sep=' ')

    #Create a dataframe with only the deepfake examples
    df = df[~df['relative_path'].str.startswith('o')]

    #Create a dataframe with all of the videos
    unique_videos=lambda x: '/'.join((x.split('/')[:-1]))
    df_unique=df['relative_path'].apply(unique_videos).drop_duplicates(ignore_index=True)

    #Produce the dataframe with the sampled video frames
    df_final=pd.DataFrame()
    #For every video
    for video in df_unique:
        #Count the number of video frames
        count=df['relative_path'].str.contains(video).sum()
        #Compute the step based on the number of frames and the sampling rate (minimum step=1)
        step=max(round(count/samples),1)
        #Compute the indexes (keep only the number of samples we want)
        rows_index=[*range(0,count,step)][:samples]
        #If the number of resulting samples is smaller than desired
        if(len(rows_index)<samples and count>=samples):
            #Sample more starting from the begining of the indexes and shifting them by one
            for i in range(samples-len(rows_index)):
                rows_index.append(rows_index[i]+1)
            rows_index.sort()
        #Find the indexes of the sampled frames in the dataframe and concacenate them to the final dataframe
        indexes = df.index[df['relative_path'].str.contains(video)].tolist()
        df_final=pd.concat([df_final,df.iloc[[indexes[i] for i in rows_index]]])

    #Save the produced dataframe and return the path
    df_final.to_csv(save_path,sep=' ',index=False)
    return save_path

#Create the csv file with the paths of the deepfake and the corresponding real examples

#ds_path: Path of the original test csv
#return: Path of the created csv

def getMatchedFFPath(ds_path):
    save_path = ds_path.split('/')
    save_path[-1] = save_path[-1].split('.')[0] + '_matched.csv'
    save_path = '/'.join(save_path)

    #Return the path if the csv already exists
    if(os.path.isfile(save_path)):
        return save_path

    #Read the original test csv
    df = pd.read_csv(ds_path, sep=' ')

    #Create an id for each example based on the video name and frame number
    video_name=lambda x: ((x.split('/')[-2]).split('_')[0]).split('.')[0]
    frame_name=lambda x: (x.split('/')[-1]).split('.')[0]
    df['id'] = df['relative_path'].apply(video_name) + '_' + df['relative_path'].apply(frame_name)

    #Create two dataframes with only the deepfake and the original examples accodringly
    df1 = df[~df['relative_path'].str.startswith('o')]
    df2 = df[df['relative_path'].str.startswith('o')]
    #Merge them based on the id
    df = pd.merge(df1,df2,on='id')

    #Spit each pair into two rows
    split_rows=[]
    for _,row in df.iterrows():
        #row1 is the path of the deepfake example
        row1 = {'relative_path': row['relative_path_x'], 'bin_label' : row['bin_label_x'], 'mc_label' : row['mc_label_x']}; split_rows.append(row1)
        #row2 is the path of the original example
        row2 = {'relative_path': row['relative_path_y'], 'bin_label' : row['bin_label_y'], 'mc_label' : row['mc_label_y']}; split_rows.append(row2)

    #Save the produced dataframe and return the path
    pd.DataFrame(split_rows).to_csv(save_path,sep=' ',index=False)
    return save_path