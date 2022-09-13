import random
import torchaudio.datasets
import torchaudio.transforms as T
import torch

dataset_labels = {
    'SPEECHCOMMANDS': {label: i for i, label in enumerate(
        ['up', 'yes', 'learn', 'visual', 'down', 'zero', 'left', 'sheila', 'eight', 'bed', 'forward', 'three', 'wow', 'happy', 'off', 'four', 'dog', 'nine', 'tree', 'five', 'marvin', 'six', 'right', 'seven', 'on', 'backward', 'house', 'no', 'cat', 'stop', 'one', 'follow', 'bird', 'two', 'go']
    )}
}

def make_collate_fn(dataset_name, sample_rate, max_sec):
    def make_collate_fn(batch):
        max_len = int(sample_rate * max_sec)
        aud_tensors = []
        labels = []
        label_to_index = dataset_labels[dataset_name]
        
        for aud_tensor, _, label, *_ in batch:
            transformed_aud_tensor = T.Resample(aud_tensor.shape[1], sample_rate)(aud_tensor)
            transformed_aud_tensor = equate_tensor_len(aud_tensor, max_len)
            aud_tensors.append(transformed_aud_tensor)
            labels.append(label_to_index[label])
            
        aud_tensors = torch.stack(tuple(aud_tensors))
        return (aud_tensors, labels)
    
    return make_collate_fn

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

if __name__=='__main__':
    train_set = torchaudio.datasets.SPEECHCOMMANDS('./backend/audio_data_uploads', subset='training', download=True)
    if train_set[0][0].shape[0] > 1:
        raise ValueError('DLP currently supports only single channel datasets')
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=41,
        collate_fn=make_collate_fn('SPEECHCOMMANDS', 16000, 0.9),
        shuffle=False
    )
    
    batch = next(iter(train_loader))
    print(batch)
    # print(train_set[0][0])
    # print(train_set[0][0].unsqueeze(0))
    # print(train_set[39], train_set[39][0].shape)
    # print(train_set[40], train_set[40][0].shape)
    # batch = [train_set[i] for i in range(40)]
    # print(batch)
    # print(make_batch_equate_len(16000, 0.9)(batch))
    