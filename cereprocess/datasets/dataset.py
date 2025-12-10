"""dataset includes the classes and functions necessary to load and preprocess the dataset.

"""
import mne
import csv
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import os
from .getfiles import get_files
from .pipeline import Pipeline, MultiPipeline, CropData, PaddedCropData
from datetime import datetime


import traceback


# Constant for normal file processing duration (5 minutes in seconds)
NORMAL_FILE_DURATION_SECONDS = 300


"""
    V2: Important changes in term of modularity to the dataset class.
    - Very important to make the dataset more modular for usage in different forms.
    - Attempt to make V2 cross-compatible with V1 of the dataset class
    - V2 is important in order to apply efficient cross-validation
    - V2 additional features include having the option to divide data into both training
        and eval splits or just one of them
"""
class Dataset:
    ''' Dataset class stores the EEG data, defining a preprocessing pipeline and converting it to other forms accordingly.
        Keeps Data in EEG format and then converts it to other forms

    '''
    def __init__(self, datapath):
        ''' Constructor Function
            INPUT:
                datapath - string - path to the MNE source files
                basedir - string - the directory before the train and eval directories
        '''
        self.trainfiles, self.evalfiles = get_files(datapath)

        # Calculating the lengths of both train and eval
        trainlen = [len(self.trainfiles['normal']), len(self.trainfiles['abnormal'])]
        evallen = [len(self.evalfiles['normal']), len(self.evalfiles['abnormal'])]
        # Also getting the fullpath
        fullpath=datapath
        fullpath = fullpath.replace("/", "")
        if len(fullpath) > 10:
            fullpath = fullpath[-10:]

        self.id = fullpath + '_T' + str(trainlen) + '_E' + str(evallen)

        self.pipeline = MultiPipeline()

    def get_id(self):
        ''' Returns the ID of the dataset
        '''
        return self.id

    def set_pipeline(self, pipeline):
        ''' Adds a pipeline to the dataset
            INPUT:
                pipeline - Pipeline - the pipeline to add
        '''
        if pipeline.__class__.__name__ == 'Pipeline':
            self.pipeline = MultiPipeline([pipeline])
        elif pipeline.__class__.__name__ == 'MultiPipeline':
            self.pipeline = pipeline
        else:
            raise ValueError("Invalid Pipeline")
            

    def add_pipeline(self, pipeline):
        ''' Adds a pipeline to the dataset
            INPUT:
                pipeline - Pipeline - the pipeline to add
        '''
        self.pipeline = self.pipeline + pipeline

    def save_all(self, destdir):
        ''' Saves the data to a numpy file
            INPUT:
                destdir - string - the directory to save the data
        '''
        os.makedirs(destdir, exist_ok=True)
        print("Saving Data to Numpy Files")

        # Open csv file to check all data
        # The csv file has data stored like folder name, data id, pipeline id, sampling rate, time span
        # Using pandas
        # if it does not exist make it

        if not os.path.exists(os.path.join(destdir, 'converted.csv')):
            columns = pd.Index(['Folder Name', 'Data ID', 'Pipeline ID', 'Sampling Rate', 'Time Span', 'Total Channels'])
            converted = pd.DataFrame(columns=columns)
            converted.to_csv(os.path.join(destdir, 'converted.csv'), index=False)


        converted = pd.read_csv(os.path.join(destdir, 'converted.csv'))
        datastored = converted[(converted['Data ID'] == self.get_id()) & (converted['Pipeline ID'] == self.pipeline.get_id())]
        if len(datastored) > 0:
            print("Data Already Stored")
            return os.path.join(destdir, 'data_processed', datastored.iloc[0]['Folder Name']), datastored.iloc[0]['Sampling Rate'], datastored.iloc[0]['Time Span'], datastored.iloc[0]['Total Channels'], self.id + '_P' + self.pipeline.get_id()

        foldername = 'results' + str(len(converted))
        destdir2 = os.path.join(destdir, 'data_processed', foldername)
        destdir2 = self.save_to_npz(destdir2)

        # Saving the data to the csv file
        newentry = pd.DataFrame([[foldername, self.get_id(), self.pipeline.get_id(), self.pipeline.sampling_rate, self.pipeline.time_span, self.pipeline.channels]], columns=converted.columns)
        converted = pd.concat([converted, newentry], ignore_index=True)
        converted.to_csv(os.path.join(destdir, 'converted.csv'), index=False)
        return destdir2, self.pipeline.sampling_rate, self.pipeline.time_span, self.pipeline.channels, foldername

    """
        Changes in V2.0
        - Save_to_npz now use pandas to store the dataset in order to get access of the dataset better :)
        - This also helps in combining the train and val datasets and storing both their annotations in the same file
        - I have also removed the separation between calling the function for train and eval datasets separately, instead the function will only be called once
    """
    def save_to_npz(self, destdir):
        """
        Saves the data to a numpy file.

        Args:
            destdir (str): The directory to save the data.
            div (str): The division of the dataset to save ('train' or 'eval').

        Returns:
            str: The destination directory where data is saved.
        """
        os.makedirs(destdir, exist_ok=True)

        # Checking the current division to convert

        records = {"all": []}
        for div, currfiles in [['train', self.trainfiles], ['eval', self.evalfiles]]:
            print(f"Converting {div} files")

            # Making the respective directory
            destdir_withdiv = os.path.join(destdir, div)
            os.makedirs(destdir_withdiv, exist_ok=True)

            # Getting the files
            normal_files = currfiles['normal']
            abnormal_files = currfiles['abnormal']

            records[div] = []

            total_pipelines = len(self.pipeline)
            print(f"Total Pipelines: {total_pipelines}")
            print(f"Total Normal Files: {len(normal_files)}")
            print(f"Total Abnormal Files: {len(abnormal_files)}")

            # Process normal and abnormal files
            print("Converting Normal Files")
            self._process_files(normal_files, [1, 0], 0, records[div], destdir_withdiv)

            print("Converting Abnormal Files")
            self._process_files(abnormal_files, [0, 1], 1, records[div], destdir_withdiv)

            records["all"] += records[div]
            # Save the records to a CSV using Pandas
            csv_path = os.path.join(destdir_withdiv, 'data.csv')
            df = pd.DataFrame(records[div])
            df.to_csv(csv_path, index=False)
            print(f"CSV saved to {csv_path}")

        # Save the records to a CSV using Pandas
        csv_path = os.path.join(destdir, 'data.csv')
        df = pd.DataFrame(records["all"])
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")
        return destdir

    def _process_files(self, files, label, label_index, records, destdir):
        for file in tqdm(files, desc=f"Processing {'Normal' if label_index == 0 else 'Abnormal'} files"):
            # ---- RAW LOAD BLOCK ----
            try:
                data = mne.io.read_raw_edf(file, preload=True, verbose='error')
                duration = data.n_times / data.info['sfreq']

            except Exception as e:
                print("\n[ERROR in raw EDF load]")
                print("File:", file)
                traceback.print_exc()
                continue

            # ---- PIPELINE BLOCK ----
            for i, pipeline in enumerate(self.pipeline):
                try:
                    # Check if the pipeline has a defined time_span (from CropData/PaddedCropData)
                    # Note: time_span is -1 by default when no CropData is present
                    if pipeline.time_span != -1:
                        time_span = pipeline.time_span
                        
                        # For normal files (label_index == 0), only keep first 5 minutes
                        if label_index == 0:
                            # Crop to first 5 minutes
                            first_5min_data = data.copy()
                            # Crop to the minimum of file duration and 5 minutes
                            crop_time = min(duration, NORMAL_FILE_DURATION_SECONDS)
                            first_5min_data.crop(tmin=0, tmax=crop_time, include_tmax=False)
                            
                            # Apply the rest of the pipeline (excluding CropData/PaddedCropData)
                            for func in pipeline.pipeline:
                                if not isinstance(func, (CropData, PaddedCropData)):
                                    first_5min_data = func.func(first_5min_data)
                            
                            processed_data = np.array(first_5min_data.get_data())
                            
                            # Create filename without segment index for normal files
                            appendname = f"_P{i}"
                            filename = f"{os.path.splitext(os.path.basename(file))[0]}{appendname}.npz"
                            np.savez(os.path.join(destdir, filename), data=processed_data, label=np.array(label))
                            
                            records.append({'File': os.path.join(destdir, filename), 'Label': label_index})
                        else:
                            # For abnormal files (label_index == 1), create segments
                            # Calculate how many complete segments we can create
                            num_segments = int(duration // time_span)
                            
                            # If no complete segments, skip this file
                            if num_segments == 0:
                                print(f"\nSkipping {file}: duration {duration}s < time_span {time_span}s")
                                continue
                            
                            # Process each segment
                            for segment_idx in range(num_segments):
                                # Calculate start and end times for this segment
                                start_time = segment_idx * time_span
                                end_time = start_time + time_span
                                
                                # Create a copy of the data for this segment
                                segment_data = data.copy()
                                segment_data.crop(tmin=start_time, tmax=end_time, include_tmax=False)
                                
                                # Apply the rest of the pipeline (excluding CropData/PaddedCropData)
                                for func in pipeline.pipeline:
                                    if not isinstance(func, (CropData, PaddedCropData)):
                                        segment_data = func.func(segment_data)
                                
                                processed_data = np.array(segment_data.get_data())
                                
                                # Create filename with segment index
                                appendname = f"_P{i}_S{segment_idx}"
                                filename = f"{os.path.splitext(os.path.basename(file))[0]}{appendname}.npz"
                                np.savez(os.path.join(destdir, filename), data=processed_data, label=np.array(label))
                                
                                records.append({'File': os.path.join(destdir, filename), 'Label': label_index})
                    else:
                        # No time_span defined, process the whole file as before
                        appendname = f"_P{i}"
                        processed_data = pipeline.apply(data)
                        processed_data = np.array(processed_data.get_data())

                        filename = f"{os.path.splitext(os.path.basename(file))[0]}{appendname}.npz"
                        np.savez(os.path.join(destdir, filename), data=processed_data, label=np.array(label))

                        records.append({'File': os.path.join(destdir, filename), 'Label': label_index})

                except Exception as e:
                    print("\n[ERROR in pipeline processing]")
                    print("File:", file)
                    print("Pipeline index:", i)
                    traceback.print_exc()
                    continue
