from dataset.cxr_dataset import CXR_dataset
from dataset.ehr_dataset import EHR_dataset
from dataset.note_dataset import Note_dataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import random

class MUL_dataset(Dataset):
    def __init__(self, split='train', img_transform=None, task='mortality', longstay_mintime=0):
        super(MUL_dataset, self).__init__()

        print('EHR dataset process...')
        self.ehr = EHR_dataset(split='all', regenerate=False, reprocess=False, task=task, longstay_mintime=longstay_mintime)
        print('CXR dataset process...')
        self.cxr = CXR_dataset(split='all', regenerate=False, transform=img_transform, task=task)
        print('Note dataset process...')
        self.note = Note_dataset(split='all', regenerate=False, task=task)
        datatable_ehr = self.ehr.get_samplelist()[['subject_id', 'hadm_id']]
        datatable_cxr = self.cxr.get_samplelist()[['subject_id', 'hadm_id', 'label']]
        datatable_note = self.note.get_samplelist()[['subject_id', 'hadm_id']]

        print("\n[EHR] shape:", datatable_ehr.shape)
        print("\n[CXR] label count:", datatable_cxr['label'].value_counts())
        print("\n[NOTE] shape:", datatable_note.shape)

        self.data_table = pd.merge(datatable_cxr, datatable_ehr, how='inner', on=['subject_id', 'hadm_id'])
        self.data_table = pd.merge(self.data_table, datatable_note, how='inner', on=['subject_id', 'hadm_id'])

        print("\nMerged data_table shape:", self.data_table.shape)

        # balance dataset
        self.label = self.data_table['label']

        df = self.data_table.copy()
        minority = df[df['label'] == 1]
        majority = df[df['label'] == 0].sample(n=len(minority), random_state=42)  # undersample
        balanced_df = pd.concat([minority, majority]).sample(frac=1, random_state=42)  # shuffle

        self.data_table = balanced_df[['subject_id', 'hadm_id']]
        self.label = balanced_df['label'].to_numpy()

        print('\nbalance data:', self.label)

        # self.label = pd.merge(self.data_table, self.ehr.all_adm_disch, how='left', on=['subject_id', 'hadm_id'])
        # self.label = self.label.apply(lambda x: (1 if x['discharge_location']=='DIED' else 0) if not x.empty else None, axis=1)
        
        # self.label = self.data_table.pop('label')
        # print("\nLabel count after merge:\n", pd.Series(self.label).value_counts())
        print('\nLabel count after balance:\n', pd.Series(self.label).value_counts())
        
        if split != 'all':
            # X_train, X_test, Y_train, Y_test = train_test_split(np.arange(len(self.label)).reshape(-1,1), self.label.to_numpy(), test_size=0.25, random_state=0)
            # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=0)
            X_train, X_test, Y_train, Y_test = train_test_split(
                np.arange(len(self.label)).reshape(-1,1),
                # self.label.to_numpy(),
                self.label,
                test_size=0.20,
                random_state=0,
                # stratify=self.label.to_numpy()
                stratify=self.label
            )

            # Split train into train + val, stratify by Y_train
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train,
                Y_train,
                test_size=0.20,
                random_state=0,
                stratify=Y_train
            )
            if split == 'train':
                # self.data_table, self.label = self.data_table.iloc[X_train.reshape(-1)], Y_train
                self.data_table = self.data_table.iloc[X_train.reshape(-1)].reset_index(drop=True)
                self.label = Y_train
            elif split == 'test':
                # self.data_table, self.label = self.data_table.iloc[X_test.reshape(-1)], Y_test
                self.data_table = self.data_table.iloc[X_test.reshape(-1)].reset_index(drop=True)
                self.label = Y_test
            if split == 'val':
                # self.data_table, self.label = self.data_table.iloc[X_val.reshape(-1)], Y_val
                self.data_table = self.data_table.iloc[X_val.reshape(-1)].reset_index(drop=True)
                self.label = Y_val

        else:
            self.label = self.label.to_numpy()
            # if isinstance(self.label, pd.Series):
            #     self.label = self.label.to_numpy()
        pass

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, index):
        x = self.data_table.iloc[index]
        subject_id = x.subject_id
        hadm_id = x.hadm_id

        demo, ce_ts, le_ts, pe_ts, timestamps = self.ehr.query(subject_id, hadm_id)
        img_list, img_position, img_time = self.cxr.query(subject_id, hadm_id)
        notes = self.note.query(subject_id, hadm_id)

        target = self.ehr.all_adm_disch[(self.ehr.all_adm_disch['subject_id'] == subject_id) &
                                        (self.ehr.all_adm_disch['hadm_id'] == hadm_id)]
        target = 1 if target['discharge_location'].iloc[0] == 'DIED' else 0

        target = self.label[index]
        return (demo, ce_ts, le_ts, pe_ts, timestamps), (img_list, img_position, img_time), notes, target
        # return (demo, ce_ts, le_ts, pe_ts, timestamps), (img_list, img_position, img_time), target
    
    def get_collate(self):
        def ehr_collate(samples):
            demo, ce_ts, le_ts, pe_ts, timestamps = map(list, zip(*map(list, samples)))
            return torch.from_numpy(np.stack(demo)), ce_ts, le_ts, pe_ts, timestamps
        def cxr_collate(samples):
            img_list, img_positions, img_times = map(list, zip(*map(list, samples)))
            return img_list, img_positions, img_times
        def note_collate(samples):
            return samples
        def collate_fn(samples):
            ehr_samples, cxr_samples, note_samples, targets = map(list, zip(*map(list, samples)))
            return ehr_collate(ehr_samples), cxr_collate(cxr_samples), note_collate(note_samples), torch.LongTensor(targets)
        return collate_fn
    
    # def get_collate(self):
    #     def ehr_collate(samples):
    #         demo, ce_ts, le_ts, pe_ts, timestamps = map(list, zip(*map(list, samples)))
    #         return torch.from_numpy(np.stack(demo)), ce_ts, le_ts, pe_ts, timestamps
    #     def cxr_collate(samples):
    #         img_list, img_positions, img_times = map(list, zip(*map(list, samples)))
    #         return img_list, img_positions, img_times
    #     def collate_fn(samples):
    #         ehr_samples, cxr_samples, targets = map(list, zip(*map(list, samples)))
    #         return ehr_collate(ehr_samples), cxr_collate(cxr_samples), torch.LongTensor(targets)
    #     return collate_fn



if __name__ == '__main__':
    import torchvision.transforms as transforms
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(norm_mean, norm_std),
    ])
    x = MUL_dataset(img_transform=img_transform,split='all' ,task='readmission')
    for i in range(len(x)):
        a = x[i]