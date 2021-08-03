import re
import os
import math
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

EPOCHS = 5
BATCH_SIZE = 2048

def train_test_model_demo(model, device, train_data_path, test_data_path, feat_dict_):
    print("Start Training Model!")

    # Sort the Train files in order
    train_filelist = ["%s%s" % (train_data_path, x) for x in os.listdir(train_data_path)]
    train_file_id = [int(re.sub('^.*[\D]', '', x)) for x in train_filelist]
    train_filelist = [train_filelist[idx] for idx in np.argsort(train_file_id)]

    # Sort the Test files in order
    test_filelist = ["%s%s" % (test_data_path, x) for x in os.listdir(test_data_path)]
    test_file_id = [int(re.sub('^.*[\D]', '', x)) for x in test_filelist]
    test_filelist = [test_filelist[idx] for idx in np.argsort(test_file_id)]

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, EPOCHS + 1):
        train_model(model, train_filelist, feat_dict_, device, optimizer, epoch)
        test_model(model, test_filelist, feat_dict_, device)


""" ************************************************************************************ """
"""                      Using Criteo DataSet to train/test Model                        """
""" ************************************************************************************ """
def train_model(model, train_filelist, feat_dict_, device, optimizer, epoch,
                use_reg_l1=False, use_reg_l2=False):
    """
    :param model:
    :param train_filelist:
    :param feat_dict_:
    :param device:
    :param optimizer:
    :param epoch:
    :param use_reg_l1:
    :param use_reg_l2:
    :return:
    """
    fname_idx = 0
    features_idxs, features_values, labels = None, None, None
    #Tong so dong du lieu cua tap train
    train_item_count = count_in_filelist_items(train_filelist)

    pre_file_data_count = 0
    #Duyet tung batch
    for batch_idx in range(math.ceil(train_item_count / BATCH_SIZE)):
        st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE #Khi batch_idx=0 thì st = 0 và ed=2048
        ed_idx = min(ed_idx, train_item_count - 1)

        if features_idxs is None:
            #Trả vè index, giá trị từng dòng, nhãn/ features_index có shape = (204800, 39)
            features_idxs, features_values, labels = get_idx_value_label(train_filelist[fname_idx], feat_dict_)
        st_idx -= pre_file_data_count
        ed_idx -= pre_file_data_count

        #khi ed < 204800
        if ed_idx < len(features_idxs):
            #lấy 2048 dòng đang xét
            batch_fea_idxs = features_idxs[st_idx:ed_idx, :]
            batch_fea_values = features_values[st_idx:ed_idx, :]
            batch_labels = labels[st_idx:ed_idx, :]
        else:
            pre_file_data_count += len(features_idxs)
            batch_fea_idxs_part1 = features_idxs[st_idx::, :]
            batch_fea_values_part1 = features_values[st_idx::, :]
            batch_labels_part1 = labels[st_idx::, :]

            fname_idx += 1
            ed_idx -= len(features_idxs)
            features_idxs, features_values, labels = get_idx_value_label(train_filelist[fname_idx], feat_dict_)
            batch_fea_idxs_part2 = features_idxs[0:ed_idx, :]
            batch_fea_values_part2 = features_values[0:ed_idx, :]
            batch_labels_part2 = labels[0:ed_idx, :]

            batch_fea_idxs = np.vstack((batch_fea_idxs_part1, batch_fea_idxs_part2))
            batch_fea_values = np.vstack((batch_fea_values_part1, batch_fea_values_part2))
            batch_labels = np.vstack((batch_labels_part1, batch_labels_part2))

        batch_fea_values = torch.from_numpy(batch_fea_values)
        batch_labels = torch.from_numpy(batch_labels)

        idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
        idx = idx.to(device)
        value = batch_fea_values.to(device, dtype=torch.float32)
        target = batch_labels.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(idx, value)
        loss = F.binary_cross_entropy_with_logits(output, target)

        # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(pytorch_total_params)
        if use_reg_l1:
            for param in model.parameters():
                loss += model.reg_l1 * torch.sum(torch.abs(param))
        if use_reg_l2:
            for param in model.parameters():
                loss += model.reg_l2 * torch.sum(torch.pow(param, 2))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(idx), train_item_count,
                100. * batch_idx / math.ceil(int(train_item_count / BATCH_SIZE)), loss.item()))


def test_model(model, test_filelist, feat_dict_, device):
    """
    :param model:
    :param test_filelist:
    :param feat_dict_:
    :param device:
    :return:
    """
    fname_idx = 0
    pred_y, true_y = [], []
    features_idxs, features_values, labels = None, None, None
    test_loss = 0
    test_item_count = count_in_filelist_items(test_filelist)
    with torch.no_grad():
        pre_file_data_count = 0
        for batch_idx in range(math.ceil(test_item_count / BATCH_SIZE)):
            st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
            ed_idx = min(ed_idx, test_item_count - 1)

            if features_idxs is None:
                features_idxs, features_values, labels = get_idx_value_label(
                    test_filelist[fname_idx], feat_dict_, shuffle=False)
            st_idx -= pre_file_data_count
            ed_idx -= pre_file_data_count

            if ed_idx <= len(features_idxs):
                batch_fea_idxs = features_idxs[st_idx:ed_idx, :]
                batch_fea_values = features_values[st_idx:ed_idx, :]
                batch_labels = labels[st_idx:ed_idx, :]
            else:
                pre_file_data_count += len(features_idxs)
                batch_fea_idxs_part1 = features_idxs[st_idx::, :]
                batch_fea_values_part1 = features_values[st_idx::, :]
                batch_labels_part1 = labels[st_idx::, :]

                fname_idx += 1
                ed_idx -= len(features_idxs)
                features_idxs, features_values, labels = get_idx_value_label(
                    test_filelist[fname_idx], feat_dict_, shuffle=False)
                batch_fea_idxs_part2 = features_idxs[0:ed_idx, :]
                batch_fea_values_part2 = features_values[0:ed_idx, :]
                batch_labels_part2 = labels[0:ed_idx, :]

                batch_fea_idxs = np.vstack((batch_fea_idxs_part1, batch_fea_idxs_part2))
                batch_fea_values = np.vstack((batch_fea_values_part1, batch_fea_values_part2))
                batch_labels = np.vstack((batch_labels_part1, batch_labels_part2))

            batch_fea_values = torch.from_numpy(batch_fea_values)
            batch_labels = torch.from_numpy(batch_labels)

            idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
            idx = idx.to(device)
            value = batch_fea_values.to(device, dtype=torch.float32)
            target = batch_labels.to(device, dtype=torch.float32)
            output = model(idx, value, use_dropout=False)

            test_loss += F.binary_cross_entropy_with_logits(output, target)

            pred_y.extend(list(output.cpu().numpy()))
            true_y.extend(list(target.cpu().numpy()))

        print('Roc AUC: %.5f' % roc_auc_score(y_true=np.array(true_y), y_score=np.array(pred_y)))
        test_loss /= math.ceil(test_item_count / BATCH_SIZE)
        print('Test set: Average loss: {:.5f}'.format(test_loss))


def count_in_filelist_items(filelist):
    count = 0
    #Doc tung file trong filelist
    for fname in filelist:
        with open(fname.strip(), 'r') as fin:
            for _ in fin:
                count += 1
    return count


def get_idx_value_label(fname, feat_dict_, shuffle=True):
    """
    :param fname:
    :param feat_dict_:
    :param shuffle:
    :return:
    """
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)
    cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cont_max_ = [5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46, 231, 4008, 7393]
    cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]

    def _process_line(line):
        #Mỗi dòng thành 1 mảng
        features = line.rstrip('\n').split('\t')
        #Lưu index
        feat_idx = []
        #Lưu giá trị sau chuẩn hóa
        feat_value = []

        # MinMax Normalization giá trị 1-13
        for idx in continuous_range_:
            #Nếu tại đặc trưng đó ko có giá trị thì thêm 0
            if features[idx] == '':
                feat_idx.append(0)
                feat_value.append(0.0)
            #Nếu có giá trị thì thêm vị trí và giá trị sau chuẩn hóa vào feat_value (Có thể cải tiến chỗ này)
            else:
                feat_idx.append(feat_dict_[idx])
                feat_value.append((float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1])
        #Giá trị từ 14-39
        for idx in categorical_range_:
            # Nếu tại đặc trưng đó ko có giá trị thì thêm 0
            if features[idx] == '' or features[idx] not in feat_dict_:
                feat_idx.append(0)
                feat_value.append(0.0)
            #Ngược lại thêm index và giá trị 1 vào feat_value
            else:
                feat_idx.append(feat_dict_[features[idx]])
                feat_value.append(1.0)
        #Trả về index, giá trị, nhãn
        return feat_idx, feat_value, [int(features[0])]

    #gọi và thực hiện hàm liền trên
    features_idxs, features_values, labels = [], [], []
    with open(fname.strip(), 'r') as fin:
        for line in fin:
            feat_idx, feat_value, label = _process_line(line)
            features_idxs.append(feat_idx)
            features_values.append(feat_value)
            labels.append(label)


    features_idxs = np.array(features_idxs)
    features_values = np.array(features_values)
    labels = np.array(labels).astype(np.int32)
    # shuffle
    if shuffle:
        idx_list = np.arange(len(features_idxs))
        np.random.shuffle(idx_list)

        features_idxs = features_idxs[idx_list, :]
        features_values = features_values[idx_list, :]
        labels = labels[idx_list, :]
    return features_idxs, features_values, labels

