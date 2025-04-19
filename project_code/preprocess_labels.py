import pandas as pd
import os

data_dir = '/uufs/chpc.utah.edu/common/home/u1527533/covid_dataset'
metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))

def simplify_label(finding):
    if 'COVID-19' in finding:
        return 'COVID-19'
    elif 'Bacterial' in finding:
        return 'Bacterial'
    elif 'Viral' in finding:
        return 'Viral'
    else:
        return 'Other'

metadata['simple_label'] = metadata['finding'].apply(simplify_label)
print(metadata[['filename', 'finding', 'simple_label']].head())
print(metadata['simple_label'].value_counts())

# 保存新 metadata
metadata.to_csv(os.path.join(data_dir, 'metadata_simplified.csv'), index=False)
