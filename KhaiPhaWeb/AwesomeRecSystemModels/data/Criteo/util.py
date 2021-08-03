import os
import numpy
import shutil
import pickle
from collections import Counter

EACH_FILE_DATA_NUM = 204800

#Save part in raw_data folder
def get_raw_data():
    if not os.path.isdir('raw_data'):
        os.mkdir('raw_data')
    print(os.getcwd())
    fin = open('../train.txt', 'r')
    fout = open('raw_data/part-0', 'w')
    for line_idx, line in enumerate(fin):
        #Each part have 204800 line
        if line_idx % EACH_FILE_DATA_NUM == 0 and line_idx != 0:
            fout.close()
            cur_part_idx = int(line_idx / EACH_FILE_DATA_NUM)
            fout = open('raw_data/part-' + str(cur_part_idx), 'w')
        fout.write(line)

    fout.close()
    fin.close()

#Split data train/test
def split_data():
    split_rate_ = 0.9
    dir_train_file_idx_ = 'aid_data/train_file_idx.txt'
    #filelist have part1-4
    filelist_ = ['raw_data/part-%d' % x for x in range(len(os.listdir('raw_data')))]
    if not os.path.exists(dir_train_file_idx_):
        train_file_idx = list(
            numpy.random.choice(
                len(filelist_), int(len(filelist_) * split_rate_), False))
        with open(dir_train_file_idx_, 'w') as fout:
            fout.write(str(train_file_idx))
    else:
        with open(dir_train_file_idx_, 'r') as fin:
            train_file_idx = eval(fin.read())
    for idx in range(len(filelist_)):
        if idx in train_file_idx:
            shutil.move(filelist_[idx], 'train_data')
        else:
            shutil.move(filelist_[idx], 'test_data')


def get_feat_dict():
    freq_ = 10
    dir_feat_dict_ = 'aid_data/feat_dict_' + str(freq_) + '.pkl2'
    #1-13 are continuous and 14-39 are categorical
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    if not os.path.exists(dir_feat_dict_):
        # Count the number of occurrences of discrete features
        feat_cnt = Counter()
        with open('../train.txt', 'r') as fin:
            for line_idx, line in enumerate(fin):
                if line_idx % EACH_FILE_DATA_NUM == 0:
                    print('generating feature dict', line_idx / 45000000)
                features = line.rstrip('\n').split('\t')
                for idx in categorical_range_:
                    if features[idx] == '':
                        continue
                    #feat_cnt sẽ là key:value
                    feat_cnt.update([features[idx]])

        # Only retain discrete features with high frequency
        dis_feat_set = set()
        for feat, ot in feat_cnt.items():
            if ot >= freq_:
                dis_feat_set.add(feat)

        # Create a dictionary for continuous and discrete features
        feat_dict = {}  #{1:1,2:2,3:3,....,13:13}
        tc = 1
        # Continuous features
        for idx in continuous_range_:
            feat_dict[idx] = tc
            tc += 1
        # Discrete features
        cnt_feat_set = set()
        with open('../train.txt', 'r') as fin:
            #Duyet file train va doc tung dong
            for line_idx, line in enumerate(fin):
                features = line.rstrip('\n').split('\t')
                #idx chay tu 14-39
                for idx in categorical_range_:
                    if features[idx] == '' or features[idx] not in dis_feat_set:
                        continue
                    #Neu gia tri chua co trong cnt_feat_set thi them vao
                    if features[idx] not in cnt_feat_set:
                        cnt_feat_set.add(features[idx])
                        feat_dict[features[idx]] = tc   #{1:1,2:2,3:3,....,13:13, 'abc':14,'cde':15}
                        tc += 1
        # Save dictionary
        with open(dir_feat_dict_, 'wb') as fout:
            pickle.dump(feat_dict, fout)
        print('args.num_feat ', len(feat_dict) + 1)
