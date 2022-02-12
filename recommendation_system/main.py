import os
import time
import math
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from models.NCF.NCF import NCF
from models.NCF_Att.NCF_Att import NCF_Att
from models.NCF.NCF_with_user_item_info import NCF_with_U_I_info
import config
import evaluate
import data_utils


def save_best_weights(epoch, HR, NDCG, best_hr, best_ndcg, best_epoch, interval_id):
    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if args.out:
            if not os.path.exists(config.model_path):
                os.mkdir(config.model_path)
            torch.save(model, '{}{}{}.pth'.format(config.model_path, config.model, interval_id))
    return best_hr, best_ndcg, best_epoch


def test_model(test_loader):
    model.eval()
    HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
    return HR, NDCG


def train(args, train_loader, test_loader, interval_id=0):
    count, best_hr, best_ndcg, best_epoch = 0, 0, 0, 0
    for epoch in range(args.epochs):
        print(" Epoch # ", epoch)
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, target in tqdm(train_loader):
            user = user.cuda()
            item = item.cuda()

            if config.user_item_info:
                label, user_info, item_info = target['label'], target['user_info'], target['item_info']
                user_info = user_info.float().cuda()
                item_info = item_info.float().cuda()
            else:
                label = target['label']
                user_info = None
                item_info = None
            label = label.float().cuda()

            model.zero_grad()
            prediction = model(user, item, user_info, item_info)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

        # Test Model:
        HR, NDCG = test_model(test_loader)
        best_hr, best_ndcg, best_epoch = save_best_weights(epoch, HR, NDCG, best_hr, best_ndcg, best_epoch, interval_id)

    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.0,
                        help="dropout rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="batch size for training")
    parser.add_argument("--epochs",
                        type=int,
                        default=25,
                        help="training epoches")
    parser.add_argument("--top_k",
                        type=int,
                        default=10,
                        help="compute metrics@top_k")
    parser.add_argument("--factor_num",
                        type=int,
                        default=32,
                        help="predictive factors numbers in the model")
    parser.add_argument("--num_layers",
                        type=int,
                        default=3,
                        help="number of layers in MLP model")
    parser.add_argument("--num_ng",
                        type=int,
                        default=4,
                        help="sample negative items for training")
    parser.add_argument("--test_num_ng",
                        type=int,
                        default=99,
                        help="sample part of negative items for testing")
    parser.add_argument("--out",
                        default=True,
                        help="save model or not")
    parser.add_argument("--gpu",
                        type=str,
                        default="0",
                        help="gpu card ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    # ############################# Load DATASET ##########################
    if config.user_item_info:
        train_data, test_data, user_num, item_num, train_mat, user_info, item_info = data_utils.load_all()
    else:
        train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

    # ########################## CREATE MODEL #################################
    if config.user_item_info and config.pretrain:
        if config.model == 'GMF':
            GMF_model = torch.load(config.GMF_model_path)
            MLP_model = None
        elif config.model == 'MLP':
            GMF_model = None
            MLP_model = torch.load(config.MLP_model_path)
        elif config.model == "NeuMF-pre":
            GMF_model = torch.load(config.GMF_model_path)
            MLP_model = torch.load(config.MLP_model_path)
    elif config.model == 'NeuMF-pre':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    elif 'NCF_Att' in config.model:
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        MLP_model = torch.load(config.MLP_model_path)
        GMF_model = None
    else:
        GMF_model = None
        MLP_model = None

    if 'NCF_Att' in config.model:
        model = NCF_Att(user_num, item_num, args.factor_num, args.num_layers,
                        args.dropout, config.model, GMF_model, MLP_model)
    elif config.user_item_info:
        model = NCF_with_U_I_info(user_num, item_num, args.factor_num,
                                  args.num_layers, args.dropout, config.model,
                                  GMF_model, MLP_model,
                                  user_info_num=user_info.shape[1],
                                  item_info_num=item_info.shape[1])
    else:
        model = NCF(user_num, item_num, args.factor_num, args.num_layers,
                    args.dropout, config.model, GMF_model, MLP_model)
    model.cuda()
    loss_function = nn.BCEWithLogitsLoss()

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # writer = SummaryWriter() # for visualization

    # Calculate the model parameters
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    if config.window_split:
        intervals_num = len(train_data)
        for i in range(intervals_num):
            # ############################# PREPARE DATASET ##########################
            if config.user_item_info:
                train_dataset = data_utils.NCFData(train_data[i], item_num, train_mat,
                                                   args.num_ng, True, user_info, item_info)
                test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False,
                                                  user_info, item_info)
            else:
                train_dataset = data_utils.NCFData(train_data[i], item_num, train_mat, args.num_ng, True)
                test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False)
            # construct the train and test datasets
            train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

            # ########################## TRAINING #####################################
            if config.mode == "training":
                train(args, train_loader, test_loader, interval_id=i)
            elif config.mode == "testing":
                assert os.path.exists(config.NeuMF_model_path), 'lack of MLP model'
                MLP_model = torch.load(config.NeuMF_model_path)
                HR, NDCG = test_model(test_loader)
            else:
                print("This mode is not implemented !!!")
    else:
        # ############################# PREPARE DATASET ##########################
        if config.user_item_info:
            train_dataset = data_utils.NCFData(train_data, item_num, train_mat,
                                               args.num_ng, True, user_info, item_info)
            test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False,
                                              user_info, item_info)
        else:
            train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
            test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False)
        # construct the train and test datasets
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

        # ########################## TRAINING #####################################
        if config.mode == "training":
            train(args, train_loader, test_loader)
        elif config.mode == "testing":
            assert os.path.exists(config.NeuMF_model_path), 'lack of MLP model'
            MLP_model = torch.load(config.NeuMF_model_path)
            HR, NDCG = test_model(test_loader)
        else:
            print("This mode is not implemented !!!")
