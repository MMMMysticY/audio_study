import torch
from functools import partial
from src.text import load_text_encoder
from src.audio import create_transform
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Batch size will be halfed if the longest wavefile surpasses threshold
HALF_BATCHSIZE_AUDIO_LEN = 800
# Note: Bucketing may cause random sampling to be biased (less sampled for those length > HALF_BATCHSIZE_AUDIO_LEN )
HALF_BATCHSIZE_TEXT_LEN = 150


def collect_audio_batch(batch, audio_transform, mode): # bucket的情况下的维度
    """
    对batch中的audio_path处理成向量 int token处理成tensor 进行padding后按照降序返回

    :param batch: batch可能是一个list [(audio_path1, int token1), (audio_path2, int token2)] 维度[batch_size, 2] 这是没有bucket的情况
                  batch还可以是一个list的list[[(file1,txt1),(file2,txt2),...]] 维度是[bucket_size, 2] 因为一旦有bucket 那么batch size就是1
    :param audio_transform: audio_transform是对音频的处理方式 是对audio的向量化表示
    :param mode: 模式 train test等
    :return: wav文件的idx, 音频的向量化表示, 音频的长度, 音频对应的文本表示
    """
    """Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>)
       e.g. [(file1,txt1),(file2,txt2),...] """
    # 一个batch的audio的组成形式是tuples (audio_path, token)

    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]

    if type(batch[0]) is not tuple:
        batch = batch[0]
    # Make sure that batch size is reasonable
    first_len = audio_transform(str(batch[0][0])).shape[0]
    if first_len > HALF_BATCHSIZE_AUDIO_LEN and mode == 'train':
        batch = batch[:len(batch)//2]
        # 如果音频的shape第一维大于HALF_BATCHSIZE_AUDIO_LEN(在Dataset中确定过batch中是降序采集的)
        # 那么就将长度变成batch的一半 只考虑后面一半的部分 这部分wavefile比较短

    # Read batch
    file, audio_feat, audio_len, text = [], [], [], []
    with torch.no_grad():
        for b in batch:
            file.append(str(b[0]).split('/')[-1].split('.')[0]) # file是wav文件的idx
            feat = audio_transform(str(b[0])) # feat是audio向量化表示
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(torch.LongTensor(b[1])) # text是文本
    # Descending audio length within each batch
    audio_len, file, audio_feat, text = zip(*[(feat_len, f_name, feat, txt)
                                              for feat_len, f_name, feat, txt in sorted(zip(audio_len, file, audio_feat, text), reverse=True, key=lambda x:x[0])])
    # 按照音频长度降序排列

    # Zero-padding
    audio_feat = pad_sequence(audio_feat, batch_first=True) # 按照最长的向量进行padding
    text = pad_sequence(text, batch_first=True) # 对tokens进行padding
    audio_len = torch.LongTensor(audio_len)

    return file, audio_feat, audio_len, text


def collect_text_batch(batch, mode):
    """
    对batch中的文本进行处理
    :param batch: 当没有bucket的情况下 batch是多个list 每个list是一个int token [batch, seq_len*] seq_len*代表长度不定
                  当有bucket的情况下 batch是list的list
    :param mode:
    :return:
    """
    '''Collects a batch of text, should be list of list of int token 
       e.g. [txt1 <list>,txt2 <list>,...] '''

    # Bucketed batch should be [[txt1, txt2,...]]
    if type(batch[0][0]) is list: # 代表是bucketed的数据
        batch = batch[0]
    # Half batch size if input to long
    if len(batch[0]) > HALF_BATCHSIZE_TEXT_LEN and mode == 'train':
        batch = batch[:len(batch)//2]
    # Read batch
    text = [torch.LongTensor(b) for b in batch]
    # Zero-padding
    text = pad_sequence(text, batch_first=True)

    return text


def create_dataset(tokenizer, ascending, name, path, bucketing, batch_size,
                   train_split=None, dev_split=None, test_split=None):
    ''' Interface for creating all kinds of dataset'''

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriDataset as Dataset
    elif name.lower() == 'dlhlp':
        from corpus.dlhlp import DlhlpDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    if train_split is not None:
        # Training mode
        mode = 'train'
        tr_loader_bs = 1 if bucketing and (not ascending) else batch_size
        bucket_size = batch_size if bucketing and (
            not ascending) else 1  # Ascending without bucketing
        # 如果显式定义了bucketing和ascending 那么tr_loader_bs就是1 采用bucket_size进行取数据
        # 如果没有显式定义bucketing和ascending 那么tr_loader_bs就是batch_size

        # Do not use bucketing for dev set
        dv_set = Dataset(path, dev_split, tokenizer, 1)
        tr_set = Dataset(path, train_split, tokenizer,
                         bucket_size, ascending=ascending)
        # Messages to show
        msg_list = _data_msg(name, path, train_split.__str__(), len(tr_set),
                             dev_split.__str__(), len(dv_set), batch_size, bucketing)

        return tr_set, dv_set, tr_loader_bs, batch_size, mode, msg_list
    else:
        # Testing model
        mode = 'test'
        # Do not use bucketing for dev set
        dv_set = Dataset(path, dev_split, tokenizer, 1)
        # Do not use bucketing for test set
        tt_set = Dataset(path, test_split, tokenizer, 1)
        # Messages to show
        msg_list = _data_msg(name, path, dev_split.__str__(), len(dv_set),
                             test_split.__str__(), len(tt_set), batch_size, False)
        msg_list = [m.replace('Dev', 'Test').replace(
            'Train', 'Dev') for m in msg_list]
        return dv_set, tt_set, batch_size, batch_size, mode, msg_list


def create_textset(tokenizer, train_split, dev_split, name, path, bucketing, batch_size):
    ''' Interface for creating all kinds of text dataset'''
    msg_list = []

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriTextDataset as Dataset
    elif name.lower() == 'dlhlp':
        from corpus.dlhlp import DlhlpTextDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    bucket_size = batch_size if bucketing else 1
    tr_loader_bs = 1 if bucketing else batch_size
    # Do not use bucketing for dev set
    dv_set = Dataset(path, dev_split, tokenizer, 1)
    tr_set = Dataset(path, train_split, tokenizer, bucket_size)

    # Messages to show
    msg_list = _data_msg(name, path, train_split.__str__(), len(tr_set),
                         dev_split.__str__(), len(dv_set), batch_size, bucketing)

    return tr_set, dv_set, tr_loader_bs, batch_size, msg_list


def load_dataset(n_jobs, use_gpu, pin_memory, ascending, corpus, audio, text):
    ''' Prepare dataloader for training/validation'''

    # Audio feature extractor
    audio_transform, feat_dim = create_transform(audio.copy())

    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # 载入一个tokenizer 模式取决于text参数的情况mode vocab

    # Dataset (in testing mode, tr_set=dv_set, dv_set=tt_set)
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, mode, data_msg = create_dataset(
        tokenizer, ascending, **corpus)
    # 调用create_dataset

    # Collect function
    collect_tr = partial(collect_audio_batch,
                         audio_transform=audio_transform, mode=mode)
    collect_dv = partial(collect_audio_batch,
                         audio_transform=audio_transform, mode='test')
    # Shuffle/drop applied to training set only
    shuffle = (mode == 'train' and not ascending)
    drop_last = shuffle
    # 如果是train模式 就shuffle且drop_last
    # 否则不shuffle 也不droplast


    # Create data loader
    tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=shuffle, drop_last=drop_last, collate_fn=collect_tr,
                        num_workers=n_jobs, pin_memory=use_gpu)
    dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=n_jobs, pin_memory=pin_memory)
    # Messages to show
    data_msg.append('I/O spec.  | Audio feature = {}\t| feature dim = {}\t| Token type = {}\t| Vocab size = {}'
                    .format(audio['feat_type'], feat_dim, tokenizer.token_type, tokenizer.vocab_size))

    return tr_set, dv_set, feat_dim, tokenizer.vocab_size, tokenizer, data_msg


def load_textset(n_jobs, use_gpu, pin_memory, corpus, text):

    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # Dataset
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, data_msg = create_textset(
        tokenizer, **corpus)
    collect_tr = partial(collect_text_batch, mode='train')
    collect_dv = partial(collect_text_batch, mode='dev')
    # Dataloader (Text data stored in RAM, no need num_workers)
    tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=True, drop_last=True, collate_fn=collect_tr,
                        num_workers=0, pin_memory=use_gpu)
    dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=0, pin_memory=pin_memory)

    # Messages to show
    data_msg.append('I/O spec.  | Token type = {}\t| Vocab size = {}'
                    .format(tokenizer.token_type, tokenizer.vocab_size))

    return tr_set, dv_set, tokenizer.vocab_size, tokenizer, data_msg


def _data_msg(name, path, train_split, tr_set, dev_split, dv_set, batch_size, bucketing):
    ''' List msg for verbose function '''
    msg_list = []
    msg_list.append('Data spec. | Corpus = {} (from {})'.format(name, path))
    msg_list.append('           | Train sets = {}\t| Number of utts = {}'.format(
        train_split, tr_set))
    msg_list.append(
        '           | Dev sets = {}\t| Number of utts = {}'.format(dev_split, dv_set))
    msg_list.append('           | Batch size = {}\t\t| Bucketing = {}'.format(
        batch_size, bucketing))
    return msg_list
