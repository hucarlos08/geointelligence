from models.data import LandsatDataset
import torch
import h5py
 # Dictionary with the configuration for the data module
config_data_module = {
        'train_file': '/teamspace/studios/this_studio/dataset/train_data.h5',
        'test_file': '/teamspace/studios/this_studio/dataset/test_data.h5',
        'batch_size': 64,
        'num_workers': 4,
        'seed': 50,
        'split_ratio': (0.8, 0.2),
        # 'transform': {
        #     'Normalize': {  'mean': [0.5,  0.5, 0.5], 
        #                     'std':  [0.5, 0.5, 0.5]
        #                     },
        # }
    }
    # Create the HDF5DataModule from the configuration
# dataset = LandsatDataset('/teamspace/studios/this_studio/dataset/train_data.h5')
# print(len(dataset))

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)


# for batch_images, batch_labels in dataloader:
#     print(batch_images.shape, batch_labels.shape)
#     pass

with h5py.File('/teamspace/studios/this_studio/dataset/train_data.h5', 'r') as hdf:
    print(hdf['images'].chunks)  # Check the chunk size