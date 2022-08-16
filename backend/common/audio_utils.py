import random
import torchaudio.datasets
import torchaudio.transforms as T
import torch

dataset_labels = {
    'SPEECHCOMMANDS': {label: i for i, label in enumerate(
        ['up', 'yes', 'learn', 'visual', 'down', 'zero', 'left', 'sheila', 'eight', 'bed', 'forward', 'three', 'wow', 'happy', 'off', 'four', 'dog', 'nine', 'tree', 'five', 'marvin', 'six', 'right', 'seven', 'on', 'backward', 'house', 'no', 'cat', 'stop', 'one', 'follow', 'bird', 'two', 'go']
    )}
}

def make_collate_fn(dataset_name, sample_rate, new_sample_rate, max_sec):
    def make_collate_fn(batch):
        max_len = int(sample_rate * max_sec)
        aud_tensors = []
        labels = []
        label_to_index = dataset_labels[dataset_name]
        resampler = T.Resample(sample_rate, new_sample_rate)
        
        for aud_tensor, _, label, *_ in batch:
            aud_tensors.append(equate_tensor_len(aud_tensor, max_len))
            aud_tensors[-1] = resampler(aud_tensors[-1])
            labels.append(label_to_index[label])
            
        aud_tensors = torch.stack(tuple(aud_tensors))
        return (aud_tensors, labels)
    
    return make_collate_fn

def equate_tensor_len(tensor, max_len):
    num_channels, signal_len = tensor.shape
    if (signal_len < max_len):
        tensor = pad_audio_tensor(tensor, num_channels, signal_len, max_len)
    elif (signal_len > max_len):
        tensor = tensor[:,:max_len]
    return tensor
        
def pad_audio_tensor(tensor, num_channels, signal_len, new_len):
    pad_len = new_len - signal_len
    print(pad_len)
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

if __name__=='__main__':
    train_set = torchaudio.datasets.SPEECHCOMMANDS('./backend/audio_data_uploads', subset='training', download=False)
    if train_set[0][0].shape[0] > 1:
        raise ValueError('DLP currently supports only single channel datasets')
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=256,
        collate_fn=make_collate_fn('SPEECHCOMMANDS', 16000, 32000, 0.9),
        shuffle=False
    )
    
    print(next(iter(train_loader)))
    # print(train_set[39], train_set[39][0].shape)
    # print(train_set[40], train_set[40][0].shape)
    # batch = [train_set[i] for i in range(40)]
    # print(batch)
    # print(make_batch_equate_len(16000, 0.9)(batch))
    