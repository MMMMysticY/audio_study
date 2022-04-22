# 自定义dlhlp数据集

from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset

# Additional (official) text src provided 扩展的文本数据
OFFICIAL_TXT_SRC = ['librispeech-lm-norm.txt']
# Remove longest N sentence in librispeech-lm-norm.txt 去除librispeech-lm-norm.txt中最长的N个句子
REMOVE_TOP_N_TXT = 5000000
# Default num. of threads used for loading LibriSpeech 加载数据集的线程数
READ_FILE_THREADS = 4


def read_text(file):
    """Get transcription of target wave file,
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread"""
    #src_file = '-'.join(file.split('-')[:-1])+'.trans.txt'
    src_file = file.rsplit('/', 1)[0]+'/bopomo.trans.txt'
    idx = file.split('/')[-1].split('.')[0] # 取出文件名中的idx

    with open(src_file, 'r') as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                return line[:-1].split(' ', 1)[1]
                # 仅分割一次第一维是idx 第二维是text


class DlhlpDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        # List all wave files
        file_list = []
        for s in split:
            #split_list = list(Path(join(path, s)).rglob("*.flac"))
            split_list = list(Path(join(path, s)).rglob("*.wav")) # 获取path+s路径下所有的wav文件名
            assert len(split_list) > 0, "No data found @ {}".format(join(path,s))
            file_list += split_list

        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in file_list) # 使用read_text函数 通过多线程从bopomo.txt中 找出对应的文字表示
        #text = Parallel(n_jobs=-1)(delayed(tokenizer.encode)(txt) for txt in text)
        text = [tokenizer.encode(txt) for txt in text]
        # 使用tokenizer进行encode

        # Sort dataset by text length
        #file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in file_list)
        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])
        # 按照tokenizer之后的text长度进行排序

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            # 这一步是防止出界 如果index到了最后bucket_size以内 将其变为len() - bucket_size
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
            # 如果显式定义了 bucket_size 那么每次getitem就返回bucket_size长度的内容
        else:
            return self.file_list[index], self.text[index]
            # 否则返回index位置的内容

    def __len__(self):
        return len(self.file_list)


class DlhlpTextDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        self.encode_on_fly = False
        read_txt_src = []

        # List all wave files
        file_list, all_sent = [], []

        for s in split:
            if s in OFFICIAL_TXT_SRC:
                self.encode_on_fly = True
                with open(join(path, s), 'r') as f:
                    all_sent += f.readlines()
            file_list += list(Path(join(path, s)).rglob("*.wav"))
            # 从OFFICIAL_TXT_SRC中读取所有的wav文件
        assert (len(file_list) > 0) or (len(all_sent)
                                        > 0), "No data found @ {}".format(path)

        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in file_list)
        all_sent.extend(text)
        # 多线程寻找对应的文本
        del text

        # Encode text
        if self.encode_on_fly:
            self.tokenizer = tokenizer
            self.text = all_sent
        else:
            self.text = [tokenizer.encode(txt) for txt in tqdm(all_sent)]
            # encode 文本
        del all_sent

        # Read file size and sort dataset by file size (Note: feature len. may be different)
        self.text = sorted(self.text, reverse=True, key=lambda x: len(x))
        if self.encode_on_fly:
            del self.text[:REMOVE_TOP_N_TXT]

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.text)-self.bucket_size, index)
            if self.encode_on_fly:
                for i in range(index, index+self.bucket_size):
                    if type(self.text[i]) is str:
                        self.text[i] = self.tokenizer.encode(self.text[i])
            # Return a bucket
            return self.text[index:index+self.bucket_size]
        else:
            if self.encode_on_fly and type(self.text[index]) is str:
                self.text[index] = self.tokenizer.encode(self.text[index])
            return self.text[index]

    def __len__(self):
        return len(self.text)
