import random
import librosa.display
import torchaudio.datasets
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch

dataset_labels = {
    'SPEECHCOMMANDS': {label: i for i, label in enumerate([
            'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
            'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
            'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero',
        ])
    }
}

def make_collate_fn(dataset_name, sample_rate, max_sec, transform):
    def collate_fn(batch):
        max_len = int(sample_rate * max_sec)
        aud_tensors = []
        labels = []
        label_to_index = dataset_labels[dataset_name]
        
        isFirst = True
        for aud_tensor, _, label, *_ in batch:
            transformed_aud_tensor = T.Resample(aud_tensor.shape[1], sample_rate)(aud_tensor)
            transformed_aud_tensor = equate_tensor_len(aud_tensor, max_len)
            
            transformed_aud_tensor = transform(transformed_aud_tensor)
            # transformed_aud_tensor = T.MelSpectrogram()(transformed_aud_tensor)
            # transformed_aud_tensor = T.AmplitudeToDB()(transformed_aud_tensor)
            # transformed_aud_tensor = T.MFCC()(transformed_aud_tensor)
            
            if isFirst:
                librosa.display.specshow(transformed_aud_tensor.squeeze().numpy())
                plt.show()
                isFirst = False
            
            aud_tensors.append(transformed_aud_tensor)
            labels.append(label_to_index[label])
            
        aud_tensors = torch.stack(tuple(aud_tensors))
        return (aud_tensors, labels)
    
    return collate_fn

def equate_tensor_len(tensor, max_len):
    signal_len = tensor.shape[1]
    if (signal_len < max_len):
        new_tensor = pad_audio_tensor(tensor, max_len)
    elif (signal_len > max_len):
        new_tensor = tensor[:,:max_len]
    else:
        new_tensor = tensor
    return new_tensor
        
def pad_audio_tensor(tensor, new_len):
    num_channels, signal_len = tensor.shape
    pad_len = new_len - signal_len
    pad_start_len = random.randint(0, pad_len)
    pad_end_len = pad_len - pad_start_len
    
    new_tensor = []
    for i in range(num_channels):
        new_signal = torch.cat((torch.zeros(pad_start_len), tensor[i], torch.zeros(pad_end_len)))
        new_tensor.append(new_signal)
    return torch.stack(tuple(new_tensor))

# def get_unique_labels(dataset):
#     unique_labels = set()
#     for data in dataset:
#         unique_labels.add(data[2])
#     print(unique_labels)
#     return unique_labels

def create_transform(transforms):
    def transform(data):
        transformed = data
        for t in transforms:
            transformed = t(transformed)
        return transformed
    return transform

if __name__=='__main__':
    train_set = torchaudio.datasets.SPEECHCOMMANDS('./backend/audio_data_uploads', subset='training', download=False)
    if train_set[0][0].shape[0] > 1:
        raise ValueError('DLP currently supports only single channel datasets')
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=256,
        collate_fn=make_collate_fn('SPEECHCOMMANDS', 16000, 0.9, create_transform([T.MelSpectrogram(), T.AmplitudeToDB()])),
        shuffle=False
    )
    
    batch = next(iter(train_loader))
    # print(batch)
    # print(train_set[0][0])
    # print(train_set[0][0].unsqueeze(0))
    # print(train_set[39], train_set[39][0].shape)
    # print(train_set[40], train_set[40][0].shape)
    # batch = [train_set[i] for i in range(40)]
    # print(batch)
    # print(make_batch_equate_len(16000, 0.9)(batch))
    