import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
import os
import argparse
import config
from utils import supervisor, tools
from other_defenses_tool_box import BackdoorDefense
from tqdm import tqdm
import matplotlib.pyplot as plt
import random, time, stats, pickle

def plot_demos(imgs, title, K=50):
    assert K in [50, 100]
    plt.figure(figsize=(40, 20))
    for i in range(len(imgs)):
        plt.subplot(5, int(K/5), i + 1)
        plt.imshow(np.transpose(imgs[i], (1, 2, 0)))
        plt.title("i=%d" % (i+1))
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def plot_img(img, title, args=None):
    # pass
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.shape[0] == 1:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title(title)
    plt.show()

def extract_entropy(data_loader, model, num_classes, K):

    model.eval()
    class_indices = [[] for _ in range(num_classes)]
    data_entropy = [torch.zeros(1).cuda() for _ in range(num_classes)]

    with torch.no_grad():
        sid = 0
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            output = model(ins_data)
            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                b_target = ins_target[bid].item()
                ss = torch.nn.functional.softmax(output[bid], -1)
                entropy = torch.sum(-ss * torch.log2(ss))
                data_entropy[b_target] = torch.cat([data_entropy[b_target], entropy.unsqueeze(dim=0)], 0)
                b_target = ins_target[bid].item()
                class_indices[b_target].append(sid + bid)
            sid += this_batch_size

    Top_indices = [[] for _ in range(num_classes)]
    Bot_indices = [[] for _ in range(num_classes)]

    for cls in range(num_classes):
        data_entropy[cls] = data_entropy[cls][1:]

        sort_confi, index_confi = torch.sort(data_entropy[cls])  # 排序
        Top_index = index_confi[:K]
        Top_indices[cls] = np.array(class_indices[cls])[Top_index.cpu().numpy()]

        Bot_index = index_confi[-K:]
        Bot_indices[cls] = np.array(class_indices[cls])[Bot_index.cpu().numpy()]

    return Top_indices, Bot_indices

def detect_outliers(data, threshold):
    """
    使用标准差方法检测数组中异常值。
    """
    mean = np.mean(data.flatten())
    std = np.std(data.flatten())
    # 计算数据点到均值的距离，判断是否超过阈值
    outliers = np.where((np.abs(data - mean)) > threshold * std, 1, 0)

    return outliers

def cal_hist(data, bin_num=20):
    data_count = stats.mode(data)
    if data_count.count[0] < 10:
        max = np.max(data)
        min = np.min(data)
        all_dis = max - min
        buckets = [[] for _ in range(bin_num)]
        for i in data:
            buckets[int((i-min)//bin_num)].append(i)
        length = []
        for bucket in buckets:
            length.append(len(bucket))
        index = np.argmax(np.array(length))
        data_index = buckets[index]
        #找众数  如果有就用 没有就返回这些
        temp = stats.mode(data_index)
        if temp.count[0] < 5:
            print("1")
            return np.mean(data_index)
        else:
            print("2")
            return temp[0][0]
    else:
        print("3")
        return data_count[0][0]

def test_trigger(model, test_loader, mask, trigger, target_label):
    model.eval()
    clean_correct = 0
    tot = 0
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0

    with torch.no_grad():
        for data, target in test_loader:

            data = data.cuda()
            target = target.cuda()
            data = ((1 - mask) * data + mask * trigger).float()
            clean_output = model(data)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct += clean_pred.eq(target_label).sum().item()

            tot += len(target)
            this_batch_size = len(target)
            tot_loss += criterion(clean_output, target) * this_batch_size

    print('Clean ACC: {}/{} = {:.6f}, Loss: {}'.format(
        clean_correct, tot,
        clean_correct / tot, tot_loss / tot
    ))

    return clean_correct / tot, None


def main(args, model, inspection_set, num_classes, K=100, N=50, data_shape=(3,32,32)):
    S = K

    kwargs = {'num_workers': 0, 'pin_memory': True}
    inspection_loader = torch.utils.data.DataLoader(
        inspection_set, batch_size=256, shuffle=False, **kwargs)

    #extract 2*K samples

    entropy_indices_dir = os.path.join(supervisor.get_poison_set_dir(args), f"entropy_indices_{K}")
    if os.path.exists(entropy_indices_dir):
        with open(entropy_indices_dir, 'rb') as file:
            Top_indices, Bot_indices = pickle.load(file)
    else:
        Top_indices, Bot_indices = extract_entropy(inspection_loader, model, num_classes, K)
        with open(entropy_indices_dir, 'wb') as file:
            pickle.dump([Top_indices, Bot_indices], file)


    # print("extract K...")
    # Top_indices, Bot_indices = extract_entropy(inspection_loader, model, num_classes, K)

    data_suspicious = [[] for _ in range(num_classes)]
    data_benign = [[] for _ in range(num_classes)]
    for cls in range(num_classes):
        temp = torch.utils.data.Subset(inspection_set, Top_indices[cls])
        suspicious_loader = torch.utils.data.DataLoader(temp, batch_size=128, shuffle=False, **kwargs)
        temp = torch.utils.data.Subset(inspection_set, Bot_indices[cls])
        benign_loader = torch.utils.data.DataLoader(temp, batch_size=128, shuffle=False, **kwargs)

        for i, (ins_data, ins_target) in enumerate(tqdm(suspicious_loader)):
            data_suspicious[cls].append(ins_data.numpy())

        for i, (ins_data, ins_target) in enumerate(tqdm(benign_loader)):
            data_benign[cls].append(ins_data.numpy())

        data_suspicious[cls] = np.array(data_suspicious[cls][0])
        data_benign[cls] = np.array(data_benign[cls][0])

    # plot_demos(data_suspicious[0], "data_suspicious", K)
    suspicious_indices = list(np.array(Top_indices[0], dtype=int))

    poison_set_dir = supervisor.get_poison_set_dir(args)
    poison_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'))
    suspicious_indices.sort()
    poison_indices.sort()
    ss = set(suspicious_indices) & set(poison_indices)
    print(f"Backdoor Sample Extraction Accuracy: {len(ss)/K*100}% / {len(ss)} / K:{K}")

    # ---------------determine target label

    data_suspicious_avg = []
    data_benign_avg = []
    delta_avg = []
    for cls in range(num_classes):
        data_suspicious_avg.append(np.mean(data_suspicious[cls], axis=0))
        data_benign_avg.append(np.mean(data_benign[cls], axis=0))
        delta_avg.append(data_suspicious_avg[cls] - data_benign_avg[cls])

    Upsilon = []
    suspicious_label = None
    for cls in range(num_classes):
        choose_cls = [i for i in range(num_classes)]
        choose_cls.pop(cls)
        choose_cls = random.choices(choose_cls, k=2)

        Theta_s = data_suspicious_avg[cls] - data_benign_avg[choose_cls[0]]
        Theta_b = data_benign_avg[choose_cls[1]] - data_benign_avg[choose_cls[0]]

        choose_sample_indices = np.random.randint(0, K, S)
        choose_sample = data_benign[cls][choose_sample_indices]
        choose_sample_Theta_s = model(torch.tensor(np.clip(choose_sample + Theta_s, 0,1)).cuda()).argmax(dim=1)
        choose_sample_Theta_b = model(torch.tensor(np.clip(choose_sample + Theta_b, 0,1)).cuda()).argmax(dim=1)

        pi_s = np.sum(choose_sample_Theta_s.cpu().numpy() == cls)
        pi_b = np.sum(choose_sample_Theta_b.cpu().numpy() == cls)

        if pi_b:
            Upsilon.append(pi_s/pi_b)
            print(f"{cls}: {pi_s/pi_b}")
        else:
            Upsilon.append(0)

        if pi_s/pi_b > 5:
            suspicious_label = cls
            break

    if suspicious_label is not None:
        print(f"suspicious_label: {suspicious_label}")
    else:
        print("Clean dataset!")
        exit(0)

    # od_class = detect_outliers(np.array(Upsilon), threshold=2.5)
    # od_class = np.where(od_class == 1)[0]
    # if suspicious_label in od_class:
    #     print(f"suspicious_label: {suspicious_label}")
    # elif (not suspicious_label) and od_class!=[]:
    #     print("Tune parameter again!")
    # else:
    #     print("Clean dataset!")
    #     exit(0)


    plot_img(data_suspicious_avg[0] * 255, "data_suspicious_avg")
    plot_img(data_benign_avg[0] * 255, "data_benign_avg")
    plot_img(delta_avg[0] * 255, "delta_avg")

    # -------------reverse the mask and trigger
    suspicious_label_flag = "replace"

    data_backdoor = data_suspicious[suspicious_label]

    random_suspicious_index = np.random.randint(0, K, N)
    random_bacdoor_data = data_backdoor[random_suspicious_index]

    l1_distance_backdoor = np.zeros(data_shape)
    for e in range(0, N-1):
        l1_distance_backdoor += (np.abs(random_bacdoor_data[e] - random_bacdoor_data[e+1])==0)
        # l1_distance_backdoor += (np.abs(random_bacdoor_data[e] - data_suspicious_avg[suspicious_label])<0.01)
    l1_distance_backdoor = l1_distance_backdoor / N
    l1_distance_backdoor = np.mean(l1_distance_backdoor, axis=0)

    if np.max(l1_distance_backdoor) < 0.2:
        suspicious_label_flag = "blend"
        # suspicious_label_flag = "wanet"

    if suspicious_label_flag=="replace":
        print("replace")

        reversed_mask = detect_outliers(np.abs(l1_distance_backdoor), threshold=2.5)
        reversed_trigger = data_backdoor[0] * reversed_mask

        plot_img(reversed_mask*255, "reversed_mask")
        plot_img(reversed_trigger*255, "reversed_trigger")

    elif suspicious_label_flag=="blend":
        print("blend")

        data_backdoor = data_suspicious[suspicious_label]
        data_backdoor_trans = np.transpose(data_backdoor, (1, 0, 2, 3))
        data_backdoor_trans = data_backdoor_trans.reshape(data_shape[0], K, -1)
        data_backdoor_trans = np.transpose(data_backdoor_trans, (0, 2, 1))  # (3，1024, K)

        m_channel = []
        for channel in data_backdoor_trans:
            channel_min = np.min(channel, axis=1)
            channel_max = np.max(channel, axis=1)
            channel_dif = (channel_max - channel_min)
            m_channel.append(np.min(1 - channel_dif))
        mask_alpha = np.max(m_channel)

        channel_pixels = np.min(data_backdoor_trans, axis=2)
        # channel_pixels = np.min(channel_pixels, axis=0)
        # reversed_trigger = np.clip(channel_pixels / mask_alpha, 0, 1).reshape(data_shape[1:])
        reversed_trigger = np.clip(channel_pixels / mask_alpha, 0, 1).reshape(data_shape)

        plot_img(reversed_trigger*255, "reverse_trigger")
        plot_img(reversed_trigger*mask_alpha*255, "reverse_trigger")
        reversed_mask = np.ones(data_shape) * mask_alpha

        # data_backdoor = data_suspicious[suspicious_label]
        # data_backdoor_trans = np.transpose(data_backdoor, (1, 0, 2, 3))
        # data_backdoor_trans = data_backdoor_trans.reshape(data_shape[0], K, -1)
        # data_backdoor_trans = np.transpose(data_backdoor_trans, (0, 2, 1))  # (3，1024, K)
        # data_normal = data_benign[suspicious_label]
        # data_normal_trans = np.transpose(data_normal, (1, 0, 2, 3))
        # data_normal_trans = data_normal_trans.reshape(data_shape[0], K, -1)
        # data_normal_trans = np.transpose(data_normal_trans, (0, 2, 1))  # (3，1024, K)
        # m_channel = []
        # for (channel_backdoor, channel_normal) in zip(data_backdoor_trans, data_normal_trans):
        #     channel_backdoor_min = np.min(channel_backdoor, axis=1)  # 黑色的值
        #     channel_backdoor_max = np.max(channel_backdoor, axis=1)  # 白色的值
        #     channel_normal_min = np.min(channel_normal, axis=1)  # 黑色的值
        #     channel_normal_max = np.max(channel_normal, axis=1)  # 白色的值
        #     # channel_dif = (channel_backdoor_max - channel_backdoor_min) / (channel_normal_max - channel_normal_min)
        #     channel_dif = channel_backdoor_max - channel_backdoor_min
        #     # print(f"min:{np.min(1 - channel_dif)}")
        #     # channel_dif = np.delete(channel_dif, np.where(channel_dif == 1.))
        #     # plt.hist(channel_dif, bins=20)
        #     # plt.show()
        #     # channel_dif = cal_hist(channel_dif, 20)
        #     m_channel.append(np.min(1 - channel_dif))
        #
        # mask_alpha = np.max(m_channel)
        #
        # # 挨个遍历求出trigger每个像素
        # channel_pixels = np.min(data_backdoor_trans, axis=2)
        # # channel_pixels = np.min(channel_pixels, axis=0)
        # reversed_trigger = np.clip(channel_pixels / mask_alpha, 0, 1).reshape(data_shape)


    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    data_transform = transforms.Compose([
        transforms.ToTensor(),])
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,batch_size=256, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    reversed_mask = torch.tensor(reversed_mask).cuda()
    reversed_trigger = torch.tensor(reversed_trigger).cuda()
    final_asr, _ = test_trigger(model, test_set_loader, reversed_mask, reversed_trigger, 0)
    if final_asr > 0.7:
        print(f"Success! ASR of the reversed trigger:{final_asr}")
        exit(0)
    else:
        print("ASR error!!!")

    # if suspicious_label_flag=="WaNet":
    # The part of code will be released once the detailed corresponding paper is received!

    print()
    exit(0)