import torch
import torch.nn as nn
import numpy as np

import scipy.sparse as sp

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt

# ------------------- label conversion tools ------------------ ##
from torch.nn import init


def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(onehot_encoder, label_encoder, labels):
    return onehot_encoder.transform(label_encoder.transform(labels).reshape(-1, 1)).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


# ------------------ RNN utils ------------------------##
def init_gru(gru):
    if isinstance(gru, nn.GRU):
        for param in gru.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(gru, nn.GRUCell):
        for param in gru.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def pad_and_pack_sequence(input_sequence):
    # pad sequences to have same length
    input_sequence = nn.utils.rnn.pad_sequence(input_sequence, batch_first=True)
    # calculate lengths of sequences and **store in a tensor**, otherwise pytorch cannot trace correctly.
    seq_lengths = torch.LongTensor(list(map(len, input_sequence)))
    # create packed sequence        
    packed_input_sequence = nn.utils.rnn.pack_padded_sequence(input_sequence, seq_lengths, batch_first=True,
                                                              enforce_sorted=False)

    return packed_input_sequence


def batch_select_tail(batch, in_lengths):
    """ Select tensors from a batch based on the time indices.
    
    E.g.     
    batch = tensor([[[ 0,  1,  2,  3],
                     [ 4,  5,  6,  7],
                     [ 8,  9, 10, 11]],

                     [[12, 13, 14, 15],
                     [16, 17, 18, 19],
                     [20, 21, 22, 23]]])
    of size = (2, 3, 4)
    
    indices = tensor([1, 2])
    
    returns tensor([[4, 5, 6, 7],
                    [20, 21, 22, 23]])
    """
    rv = torch.stack([torch.index_select(batch[i], 0, in_lengths[i] - 1).squeeze(0) for i in range(batch.size(0))])

    return rv


def batch_mean_pooling(batch, in_lengths):
    """ Select tensors from a batch based on the input sequence lengths. And apply mean pooling over it.

    E.g.
    batch = tensor([[[ 0,  1,  2,  3],
                     [ 4,  5,  6,  7],
                     [ 8,  9, 10, 11]],

                     [[12, 13, 14, 15],
                     [16, 17, 18, 19],
                     [20, 21, 22, 23]]])
    of size = (2, 3, 4)

    indices = tensor([1, 2])

    returns tensor([[0, 1, 2, 3],
                    [14, 15, 16, 17]])
    """
    mean = []

    for idx, instance in enumerate(batch):
        keep = instance[:int(in_lengths[idx])]

        mean.append(torch.mean(keep, dim=0))

    mean = torch.stack(mean, dim=0)
    return mean


def gather_last(batch_hidden_states, in_lengths):
    num_hidden_states = int(batch_hidden_states.size(2))

    indices = in_lengths.unsqueeze(1).unsqueeze(1) - 1
    indices = indices.repeat(1, 1, num_hidden_states)

    return torch.gather(batch_hidden_states, 1, indices).squeeze(1)


# --------------- plotting utils -------------------- #

def plot_curves(A=None, B=None, C=None, D=None):
    if not A:
        A = np.load('output/epoch_training_losses.npy')
        B = np.load('output/epoch_training_scores.npy')
        C = np.load('output/epoch_test_loss.npy')
        D = np.load('output/epoch_test_score.npy')

    epochs = A.shape[0]
    # plot
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), np.mean(A, axis=1))  # train loss (on epoch end)
    plt.plot(np.arange(1, epochs + 1), C)  # test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc="upper left")

    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, epochs + 1), np.mean(B, axis=1))  # train accuracy (on epoch end)
    plt.plot(np.arange(1, epochs + 1), D)  # test accuracy (on epoch end)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc="upper left")
    title = "output/curves.png"
    plt.savefig(title, dpi=600)
    # plt.close(fig)
    # plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          save_to=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label',
           )

    ax.tick_params(axis='both', which='major', labelsize=5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, format(cm[i, j], fmt),
    #                 ha="center", va="center",
    #                 color="white" if cm[i, j] > thresh else "black",
    #                 fontdict={'weight': 'bold', 'size': 5})
    fig.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=600)

    return ax


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    # batch_hidden_states = torch.FloatTensor([[[0, 1, 2, 3],
    #                            [4, 5, 6, 7],
    #                            [8, 9, 10, 11]],
    #
    #                           [[12, 13, 14, 15],
    #                            [16, 17, 18, 19],
    #                            [20, 21, 22, 23]]])

    t = torch.Tensor([1, 2, 3])

    mask_indices = [0, 2]
    mask_indices = torch.LongTensor(mask_indices)

    # frame_preds = torch.index_select(frame_preds, dim=0, index=mask_indices)
    gt = torch.index_select(t, dim=0, index=mask_indices)

    print(gt)

    # # batch_lengths = torch.ones(size=(4, 20))
    #
    # fc2 = nn.Linear(4, 20)
    # import torch.nn.functional as F
    # loss = F.cross_entropy(batch_hidden_states[0], torch.LongTensor([[0, 1, 2], [0, 1, 2]]))
    # print(loss)
    # print(mean_pooling(batch_hidden_states, batch_lengths))
    # print(batch_mean_pooling(batch_hidden_states, batch_lengths))
    # print(fc2(batch_hidden_states).size())
    #
    # hidden_x_dirs = int(batch_hidden_states.size(2))
    #
    # indices = batch_lengths.unsqueeze(1).unsqueeze(1) - 1
    # indices = indices.repeat(1, 1, hidden_x_dirs)
    #
    # last_hidden_out = torch.gather(batch_hidden_states, 1, indices).squeeze(1)
    #
    # print(last_hidden_out)
