import pandas as pd
import os

data_dir = '/uufs/chpc.utah.edu/common/home/u1527533/covid_dataset'
metadata = pd.read_csv(os.path.join(data_dir, 'metadata_simplified.csv'))

# 僅保留 .jpeg 和 .png
valid_extensions = ['.jpeg', '.jpg', '.png']
metadata['extension'] = metadata['filename'].str.lower().str.extract(r'(\.\w+)$')[0]
filtered_metadata = metadata[metadata['extension'].isin(valid_extensions)].copy()
filtered_metadata = filtered_metadata.drop(columns=['extension'])

print(f"Original size: {len(metadata)}")
print(f"Filtered size: {len(filtered_metadata)}")
print(filtered_metadata['simple_label'].value_counts())

# 保存過濾後的 metadata
filtered_metadata.to_csv(os.path.join(data_dir, 'metadata_filtered.csv'), index=False)
