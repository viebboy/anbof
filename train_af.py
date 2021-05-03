import os, sys, getopt, pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import itertools
import trainer as model_trainer
import models

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        x = self.x[i]
        y = int(self.y[i][0])
        return torch.from_numpy(x).float(), torch.tensor([y,]).long() 


def get_AF_dataset(data_file, fold_idx, sample_length=30, standardize=True):
    assert sample_length >= 30, 'The `sample_length` of AF dataset must be at least 30 seconds'
    assert fold_idx in [0, 1, 2, 3, 4], '`fold_idx` must be from the set {0, 1, 2, 3, 4}'

    assert os.path.exists(data_file)

    # read data file
    fid = open(data_file, 'rb')
    data = pickle.load(fid)
    fid.close()

    # prepare data fold
    train_indices = data['train_indices'][fold_idx]
    test_indices = data['test_indices'][fold_idx]
    x = data['x']
    y = data['y']
    nb_train = len(train_indices)
    nb_test = len(test_indices)
    length = data['sampling_frequency'] * sample_length

    x_train = np.zeros((nb_train, 1, length), dtype=np.float32)
    y_train = np.asarray(y)[train_indices] - 1

    x_test = np.zeros((nb_test, 1, length), dtype=np.float32)
    y_test = np.asarray(y)[test_indices] - 1

    for count, idx in enumerate(train_indices):
        x_train[count, 0, :] = pad_sample(x[idx], length)

    for count, idx in enumerate(test_indices):
        x_test[count, 0, :] = pad_sample(x[idx], length)

    c0 = np.where(y_train == 0)[0].size
    c1 = np.where(y_train == 1)[0].size
    c2 = np.where(y_train == 2)[0].size
    c3 = np.where(y_train == 3)[0].size

    class_weight = [1e3/float(c0), 1e3/float(c1), 1e3/float(c2), 1e3/float(c3)]

    if standardize:
        x_tmp = x_train.flatten()
        x_tmp = x_tmp[np.where(x_tmp != 0)]
        mean = np.reshape(np.mean(x_tmp), (1, 1, 1))
        std = np.reshape(np.std(x_tmp), (1, 1, 1))
        if std[0, 0, 0] < 1e-5:
            std[0, 0, 0] = 1.0

        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

    train_set = Dataset(x_train, y_train)
    test_set = Dataset(x_test, y_test)

    return train_set, test_set, length, class_weight


def pad_sample(x, size):
    x = x.flatten()
    if x.size >= size:
        y = x[:size]
    else:
        y = np.zeros((size,), dtype=np.float32)
        y[:x.size] = x

    return y

def main(argv):
    try:
      opts, args = getopt.getopt(argv,"h", ['fold-idx=',
                                            'quantization-type=',
                                            'attention-type='])

    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--fold-idx':
            fold_idx = int(arg)
        elif opt == '--quantization-type':
            quantization_type = arg
        elif opt == '--attention-type':
            attention_type = arg
        else:
            raise RuntimeError('unknown option: {}'.format(arg))

    n_class = 4
    batch_size = 32
    sample_length = 30
    n_codeword = 512
    dropout = 0.2
    device = torch.device('cuda')


    train_set, val_set, series_length, class_weight = get_AF_dataset('AF.pickle', fold_idx, sample_length)
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader = None

    dummy_x = train_set[0][0] 
    series_length = dummy_x.numpy().size
    
    trainer = model_trainer.ClassifierTrainer(n_epoch=2, class_weight=class_weight, test_mode=True) 


    if quantization_type == 'nbof':
        model = models.ANBoF(in_channels=1,
                             series_length=series_length,
                             n_codeword=n_codeword,
                             att_type=attention_type,
                             n_class=n_class,
                             dropout=dropout)
    elif quantization_type == 'tnbof':
        model = models.ATNBoF(in_channels=1,
                              series_length=series_length,
                              n_codeword=n_codeword,
                              att_type=attention_type,
                              n_class=n_class,
                              dropout=dropout)
    else:
        raise RuntimeError('Unknown quantization type: {}'.format(quantization_type))


    performance = trainer.fit(model, train_loader, val_loader, test_loader, device=device) 
    

if __name__ == "__main__":
    main(sys.argv[1:])

