from __future__ import division
import torch
from data_reader_general import data_reader, data_generator
import pickle
import models
from util import AverageMeter
from util import save_checkpoint as save_best_checkpoint
import os.path as osp
import torch.backends.cudnn as cudnn
import argparse
from torch import optim
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
torch.manual_seed(222)

# Get model names in the folder
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


def adjust_learning_rate(optimizer, epoch, args):
    '''
    Descend learning rate
    '''
    lr = args.lr / (2 ** (epoch // args.adjust_every))
    print("Adjust lr to ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(save_model, i_iter, args, is_best=True):
    '''
    Save the model to local disk
    '''
    dict_model = save_model.state_dict()
    filename = args.snapshot_dir
    save_best_checkpoint(dict_model, is_best, i_iter, filename)


def train(model, dg_train, dg_valid, dg_test, optimizer, args):
    cls_loss_value = AverageMeter(10)
    best_acc = 0
    best_f1 = 0
    model.train()
    is_best = False
    loops = int(dg_train.data_len / args.batch_size)
    for e_ in range(args.epoch):
        print('Epoch numer: ', e_)
        dg_train.reset_samples()
        if e_ % args.adjust_every == 0:
            adjust_learning_rate(optimizer, e_, args)
        for idx in range(loops):

            sent_vecs, mask_vecs, label_list, sent_lens, _, _, _ = next(dg_train.get_ids_samples())
            if args.if_gpu:
                sent_vecs, mask_vecs = sent_vecs.cuda(), mask_vecs.cuda()
                label_list, sent_lens = label_list.cuda(), sent_lens.cuda()
            cls_loss, norm_pen = model(sent_vecs, mask_vecs, label_list, sent_lens)
            cls_loss_value.update(cls_loss.item())

            total_loss = cls_loss + norm_pen
            model.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, norm_type=2)
            optimizer.step()
        valid_acc, valid_f1 = evaluate_test(dg_valid, model, args)

        if valid_f1 > best_f1:
            open('output.txt', 'w').close()
            open('test_id.txt', 'w').close()
            is_best = True
            best_f1 = valid_f1
            save_checkpoint(model, e_, args, is_best)
            output_samples = True
            if e_ % 10 == 0:
                output_samples = True
            test_acc, test_f1 = evaluate_test(dg_test, model, args, output_samples, test=True)

        model.train()
        is_best = False

    return test_acc, test_f1


def evaluate_test(dr_test, model, args,  sample_out=False, test=False):
    mistake_samples = 'data/mistakes.txt'
    result = 'output.txt'
    with open(mistake_samples, 'w') as f:
        f.write('Test begins...')

    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    true_labels = []
    pred_labels = []
    sent_v = []
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len, texts, targets, _ = next(dr_test.get_ids_samples(test))
        sent, mask, sent_len, label = sent.cuda(), mask.cuda(), sent_len.cuda(), label.cuda()
        pred_label, best_seq, sent_vs,  score = model.predict(sent, mask, sent_len)
        # Compute correct predictions
        correct_count += sum(pred_label == label).item()
        true_labels.extend(label.cpu().numpy())
        pred_labels.extend(pred_label.cpu().numpy())
        #  Output wrong samples, for debugging
        indices = torch.nonzero(pred_label != label)

        if len(indices) > 0:
            indices = indices.squeeze(1)
        sent_vss = []
        sent_vss.extend(sent_vs.detach().cpu().numpy())
        sent_v.extend(sent_vss)
        
        if sample_out:
            with open(result, 'a') as f:
                for i in range(len(label)):
                    line = texts[i] + '###' + ' '.join(targets[i]) + '###' + str(label[i]) + '###' + str(pred_label[i]) + '\n'
                    f.write(line)
            with open(mistake_samples, 'a') as f:
                for i in indices:
                    line = texts[i] + '###' + ' '.join(targets[i]) + '###' + str(label[i]) + '###' + str(pred_label[i]) + '\n'
                    f.write(line)
    if test:
        pickle.dump(sent_v, open('sent_vs.pkl', 'wb'))
    acc = correct_count * 1.0 / dr_test.data_len
    f1 = f1_score(true_labels, pred_labels, average='macro')
    if test:
        print('test result: ', (acc, f1))
    return acc, f1

# uncomment the following line for different dataset accordingly
# def main(l_hidden_size,dropout, dropout2, mask_dim , power, batch_size, num_layer):

# 16rest embed_num = 4419 and data path accordingly
# def main(l_hidden_size=2,dropout=4, dropout2=4, mask_dim=2 , power=3, batch_size=2, num_layer=2):

# # #14rest modify embed_num = 5120 and data path accordingly
# def main(l_hidden_size=1,dropout=8, dropout2=5, mask_dim=2, power=1, batch_size=3, num_layer=2):

# #14lap embed_num = 4070 and data path accordingly
def main(l_hidden_size=2, dropout=5, dropout2=6, mask_dim=4, power=1, batch_size=3, num_layer=1):

# 15rest embed_num = 3549 and data path accordingly
# def main(l_hidden_size=1, dropout=3, dropout2=3, mask_dim=3 , power=1, batch_size=2, num_layer=2):

    parser = argparse.ArgumentParser()
    parser.add_argument('-training', type=bool, default=True)
    parser.add_argument('-embed_num', type=int, default=4070, help='The correct vocab size will print to screen, if error appears')
    parser.add_argument('-arch', type=str, default='AspectSent')
    parser.add_argument('-batch_size', type=int, default=int(32*batch_size))
    parser.add_argument('-mask_dim', type=int, default=int(mask_dim*20+10))
    parser.add_argument('-l_hidden_size', type=int, default=int(l_hidden_size * 32))
    parser.add_argument('-l_num_layers', type=int, default=int(num_layer *2))
    parser.add_argument('-l_dropout', type=int, default=0.1)
    parser.add_argument('-power', type=int, default=int(power))
    parser.add_argument('-dropout', type=int, default=dropout * 0.1)
    parser.add_argument('-dropout2',  type=int, default=dropout2 * 0.1)
    parser.add_argument('-g_num_layer', type=int, default=2)
    parser.add_argument('-embed_dim', type=int, default=300)
    parser.add_argument('-if_update_embed', type=bool, default=False)
    parser.add_argument('-if_reset', type=bool, default=True)
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-lr', type=int, default=0.008)
    parser.add_argument('-adjust_every', type=int, default=8)
    parser.add_argument('-clip_norm', type=int, default=3)
    parser.add_argument('-finetune_embed', type=bool, default=False)
    parser.add_argument('-if_gpu', type=bool, default=True)
    parser.add_argument('-use_gpu', type=bool, default=True)
    parser.add_argument('-pretrained_embed_path', type=str, default='data/glove.840B.300d.txt')
    parser.add_argument('-exp_name', type=str, default='laptop')
    parser.add_argument('-embed_path', type=str, default='data/laptop/vocab/local_emb.pkl')
    parser.add_argument('-data_path', type=str, default='data/laptop/')
    parser.add_argument('-train_path', type=str, default='data/laptop/train.pkl')
    parser.add_argument('-valid_path', type=str, default='data/laptop/valid.pkl')
    parser.add_argument('-test_path', type=str, default='data/laptop/test.pkl')
    parser.add_argument('-dic_path', type=str, default='data/laptop/vocab/dict.pkl')
    parser.add_argument('-bestmodel_path', type=str, default='checkpoints/laptop/bestmodel.pth.tar')
    parser.add_argument('-model_path', type=str, default='data/models/')
    parser.add_argument('-snapshot_dir', type=str, default='checkpoints/')

    args = parser.parse_args()
    cudnn.enabled = True
    args.snapshot_dir = osp.join(args.snapshot_dir, args.exp_name)

    global best_acc
    best_acc = 0
    # Load datasets
    dr = data_reader(args)
    train_data = dr.load_data(args.train_path)
    valid_data = dr.load_data(args.valid_path)
    test_data = dr.load_data(args.test_path)

    dg_train = data_generator(args, train_data)
    dg_valid = data_generator(args, valid_data, False)
    dg_test = data_generator(args, test_data, False)

    model = models.__dict__[args.arch](args)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    if args.if_gpu:
        model = model.cuda()
    if args.training:
        # to decide if load saved best model or not
        test_f1 = train(model, dg_train, dg_valid, dg_test, optimizer, args)
    else:
        # modify the best model path if want to use the best model to test
        model.load_state_dict(torch.load(args.bestmodel_path))
        test_acc, test_f1 = evaluate_test(dg_test, model, args, sample_out=False, test=True)

    return test_f1


if __name__ == "__main__":
    main()
