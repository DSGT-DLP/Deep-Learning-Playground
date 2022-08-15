import random
import torchaudio
import torch

def batch_equate_len(batch, sample_rate, max_sec):
    max_len = int(sample_rate * max_sec)
    new_batch = []
    
    for tensor in batch:
        num_channels, signal_len = tensor.shape
        if (signal_len < max_len):
            tensor = pad_audio_tensor(tensor, num_channels, signal_len, max_len)
        elif (signal_len > max_len):
            tensor = tensor[:,:max_len]
        new_batch.append(tensor)
    
    return torch.stack(tuple(new_batch))
        
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

if __name__=='__main__':
    train_set = torchaudio.datasets.SPEECHCOMMANDS('./backend/audio_data_uploads', subset='training', download=False)
    # datapoint = train_set[0]
    # new_datapoint = batch_equate_len(datapoint[0], 16000, 1.1)
    # print(new_datapoint)
    # print(new_datapoint.shape)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=False)
    # batch = next(iter(train_loader))[0]
    batch = torch.tensor([train_set[i][0].tolist() for i in range(39)])
    print(batch)
    print(batch_equate_len(batch, 16000, 0.9))
    