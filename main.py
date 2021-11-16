import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataloader import DatasetLoader, collate_fn
from primary_pred_module import primModel
from ancillary_pred_module import anclModel
from self_correction_module import selfcorrModel

parser = argparse.ArgumentParser()
#dataset
parser.add_argument('--dataset', type=str, default='salad', help='dataset', choices=['breakfast', 'salad'])
parser.add_argument('--feature_type', type=str, default='fisher', help='feature type, for salad, only have first three choices.', choices=['fisher', 'gt', 'I3D', 'fisher_label', 'I3D_label', 'fisher_label_cat', 'I3D_label_cat'])
parser.add_argument('--n_classes', type=int, default=19, help='action classes, corresponding to dataset', choices=[48, 19])
parser.add_argument('--observation', type=str, default='obs-0.3', help='portion of observed video', choices=['obs-0.2', 'obs-0.3'])
parser.add_argument('--prediction', type=float, default=0.1, help='portion of predicted video', choices=[0.1, 0.2, 0.3, 0.5])
parser.add_argument('--fps', type=int, default=30, help='fps of video, corresponding to dataset', choices=[15, 30])

#video preprocessing
parser.add_argument('--len_S_list', nargs='+', type=int, default=[5, 10, 15], help='S to be divided into how many clips')
parser.add_argument('--len_R', type=int, default=5, help='R to be divided into how many clips')
parser.add_argument('--startpoints_R', nargs='+', type=float, default=[5, 10, 15], help='startpoints of R (how many seconds before current time point')

#model hypermeters
parser.add_argument('--conv_dim_NLB', type=int, default=128, help='out_channel dimension of the convolution layer in NLB')
parser.add_argument('--linear_dim', type=int, default=1024, help='dimension of the linear layer in CB.')
parser.add_argument('--dropout_NLB', type=float, default=0.3, help='dropout rate of the dropout layer in NLB')
parser.add_argument('--dropout_CB', type=float, default=0.3, help='dropout rate of the dropout layer in CB')
parser.add_argument('--dropout_TAB', type=float, default=0.3, help='dropout rate of the dropout layer in TAB')
parser.add_argument('--hidden_dim_LSTM', type=int, default=512, help='hidden layer of LSTM (decoder of dense prediction)')
parser.add_argument('--max_len', type=int, default=25, help='maximum times of LSTM recurrence (should be long enough that no video has more clips to predict than this number, breakfast is 24, salad is 25.)')
parser.add_argument('--light', type=bool, default=True, help='whether to use light version model (refer to block.py for details)')

#self correction module
parser.add_argument('--self_correction_method', type=str, default='auto', help='which method to use in self correction module', choices=['no', 'linear', 'auto'])
parser.add_argument('--alpha', nargs=2, type=float, default=[30, 0.5], help='start and end value of alpha in self correction module (start>end), only needed when self correction module method is "linear"')

#other
parser.add_argument('--model', type=str, default='/model', help='path to save model')
parser.add_argument('--batch', type=int, default=2, help='batch size (salad is 2, breakfast is 16)')

args = parser.parse_args()

datapath = args.dataset + '/features/' #change to your datapath
modelpath = args.dataset + args.model  #change to your modelpath (path to save trained models)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if args.dataset == 'breakfast':
    if args.feature_type == 'gt' or args.feature_type == 'fisher_label' or args.feature_type == 'I3D_label':
        video_feat_dim = args.n_classes
    elif args.feature_type == 'fisher':
        video_feat_dim = 64
    elif args.feature_type == 'I3D':
        video_feat_dim = 400
    elif args.feature_type == 'fisher_label_cat':
        video_feat_dim = 64 + args.n_classes
    elif args.feature_type == 'I3D_label_cat':
        video_feat_dim = 400 + args.n_classes
else: #args.dataset == 'salad'
    if args.feature_type == 'gt':
        video_feat_dim = args.n_classes
    elif args.feature_type == 'fisher':
        video_feat_dim = 64
    elif args.feature_type == 'I3D':
        video_feat_dim = 2048
        
def mycrossentropy(prob, gt):
    loss = 0
    prob = F.softmax(prob, 1)
    for i in range(len(prob)):
        loss -= torch.sum(gt[i]*torch.log(prob[i]))
    return loss
        
def main():
    alpha = args.alpha[0]
    end = args.alpha[1]
    full = 8 #how many data in full set
    total = 40 #how many data in training set (including full set and weak set)
    light = 'light' if args.light else 'heavy'
    
    anci = anclModel(args, video_feat_dim).to(device)
    prim = primModel(args, video_feat_dim).to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    loss_mse = nn.MSELoss(reduction='sum')
    
    if not args.self_correction_method == 'auto':
        #step1: train ancillary model using full set
        anci.train()
        optimizer1 = optim.Adam(anci.parameters(), lr=0.001, betas=(0.99, 0.9999))
        scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[5, 15])
        print('-------Start training ancillary model-------')
        
        for e in range(20):
            s = time.time()
            fullset = DataLoader(dataset=DatasetLoader(args, datapath, 'full'), batch_size=args.batch, shuffle='True',
                                 collate_fn=collate_fn)
            total_loss = []
            total_acc = 0
            n = 0
            
            for S, R, fl, wl, dl in fullset:
                loss = 0
                optimizer1.zero_grad()
                label, prob, curr_action, pred_action_durations, _ = anci(S, R, wl)
                
                loss += loss_fn(curr_action, wl)
                for i in range(prob.shape[0]):
                    loss += loss_mse(pred_action_durations[i][:len(dl[i])-1], dl[i][:-1])
                    if len(fl[i]) >0:
                        loss += loss_fn(prob[i][:len(fl[i])], fl[i])
                loss.backward()
                
                optimizer1.step()
                total_loss.append(loss)

                label = torch.stack(label).cpu().numpy().T #batch_size*max_len
                for i in range(len(fl)):
                    gt_frame = [int(wl[i])] * int(dl[i][0] * dl[i][-1])
                    pred_frame = [int(torch.argmax(curr_action[i]))] * int(pred_action_durations[i][0] * dl[i][-1])
                    for j in range(1, len(dl[i])-1):
                        gt_frame.extend([int(fl[i][j-1])] * int(dl[i][j] * dl[i][-1]))
                        pred_frame.extend([int(label[i][j-1])] * int(pred_action_durations[i][j] * dl[i][-1]))
                        
                    min_len = min(len(gt_frame), len(pred_frame))
                    if min_len > 0:
                        n += 1
                        total_acc += accuracy_score(gt_frame[:min_len], pred_frame[:min_len])
             
            acc = total_acc/n if n > 0 else 0     
            scheduler1.step()
            print('step1 epoch %d: average loss is %.4f, total time %.2s seconds, acc %.4f' % (e+1, sum(total_loss)/full, time.time()-s, acc))
                
        #step2: train primary model using full set and weak set with ancillary model fixed
        optimizer2 = optim.Adam(prim.parameters(), lr=0.001, betas=(0.99, 0.9999))
        scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[3, 15])
        anci.eval()
        prim.train()
        
        print('-------Start training primary model-------')
        
        for e in range(25):
            s = time.time()
            fullset = DataLoader(dataset=DatasetLoader(args, datapath, 'full'), batch_size=args.batch, shuffle='True', 
                                 collate_fn=collate_fn)
            weakset = DataLoader(dataset=DatasetLoader(args, datapath, 'weak'), batch_size=args.batch*2, shuffle='True', 
                                 collate_fn=collate_fn)
            total_loss = []
            total_acc = 0
            n = 0
            
            for S, R, fl, wl, dl in fullset:
                loss = 0
                optimizer2.zero_grad()
                label, prob, curr_action, pred_action_durations, _ = prim(S, R)
                
                loss += loss_fn(curr_action, wl)
                for i in range(prob.shape[0]):
                    loss += loss_mse(pred_action_durations[i][:len(dl[i])-1], dl[i][:-1])
                    if len(fl[i]) >0:
                        loss += loss_fn(prob[i][:len(fl[i])], fl[i])
                loss.backward()
                
                optimizer2.step()
                total_loss.append(loss)

                label = torch.stack(label).cpu().numpy().T #batch_size*max_len
                for i in range(len(fl)):
                    gt_frame = [int(wl[i])] * int(dl[i][0] * dl[i][-1])
                    pred_frame = [int(torch.argmax(curr_action[i]))] * int(pred_action_durations[i][0] * dl[i][-1])
                    for j in range(1, len(dl[i])-1):
                        gt_frame.extend([int(fl[i][j-1])] * int(dl[i][j] * dl[i][-1]))
                        pred_frame.extend([int(label[i][j-1])] * int(pred_action_durations[i][j] * dl[i][-1]))
                        
                    min_len = min(len(gt_frame), len(pred_frame))
                    if min_len > 0:
                        n += 1
                        total_acc += accuracy_score(gt_frame[:min_len], pred_frame[:min_len])
             
            acc = total_acc/n if n > 0 else 0
              
            for S, R, _, wl, _ in weakset: 
                loss = 0
                optimizer2.zero_grad()
                
                _, prob_p, curr_action, pred_action_durations_p, attention_p = prim(S, R)
                with torch.no_grad():
                    _, prob_a, _, pred_action_durations_a, attention_a = anci(S, R, wl)
                if args.self_correction_method == 'no':
                    sfl = prob_a
                    sfad = pred_action_durations_a
                else:
                    corr = selfcorrModel(args, alpha)
                    sfl, sfad = corr(prob_p, prob_a, pred_action_durations_p, pred_action_durations_a)
                    
                loss += loss_fn(curr_action, wl)
                loss += loss_mse(pred_action_durations_p, sfad)
                for i in range(args.max_len):
                    loss += torch.norm(attention_a[i]-attention_p[i], p=2)
                for i in range(prob_p.shape[0]):
                    loss += mycrossentropy(prob_p[i], sfl[i])
                loss.backward()
                optimizer2.step()
                total_loss.append(loss)
                
            scheduler2.step()
            alpha = max(alpha*0.95, end)
            print('step2 epoch %d: average loss is %.4f, total time %.2s seconds, acc %.4f' % (e+1, sum(total_loss)/total, time.time()-s, acc))
        
        #step3: test
        prim.eval()
        with torch.no_grad():
            testset = DataLoader(dataset=DatasetLoader(args, datapath, 'test'), batch_size=args.batch, shuffle='False', 
                                 collate_fn=collate_fn)
            total_acc = 0
            n = 0
           
            for S, R, fl, wl, dl in testset:
                label, _, curr_action, pred_action_durations, _ = prim(S, R)
                label = torch.stack(label).cpu().numpy().T #batch_size*max_len
                for i in range(len(fl)):
                    gt_frame = [int(wl[i])] * int(dl[i][0] * dl[i][-1])
                    pred_frame = [int(torch.argmax(curr_action[i]))] * int(pred_action_durations[i][0] * dl[i][-1])
                    for j in range(1, len(dl[i])-1):
                        gt_frame.extend([int(fl[i][j-1])] * int(dl[i][j] * dl[i][-1]))
                        pred_frame.extend([int(label[i][j-1])] * int(pred_action_durations[i][j] * dl[i][-1]))
                        
                    min_len = min(len(gt_frame), len(pred_frame))
                    if min_len > 0:
                        n += 1
                        total_acc += accuracy_score(gt_frame[:min_len], pred_frame[:min_len])
             
            acc = total_acc/n if n > 0 else 0
            print('frame-wise accuracy on test set is %.4f' % acc)
            
    else:
        #step1: train ancillary model using half full set
        anci.train()
        optimizer1 = optim.Adam(anci.parameters(), lr=0.001, betas=(0.99, 0.9999))
        scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[5])
        print('-------Start training ancillary model-------')
        
        for e in range(15):
            s = time.time()
            fullset = DataLoader(dataset=DatasetLoader(args, datapath, 'full', half=True), batch_size=int(args.batch/2), shuffle='True',
                                 collate_fn=collate_fn)
            total_loss = []
            total_acc = 0
            n = 0
            
            for S, R, fl, wl, dl in fullset:
                loss = 0
                optimizer1.zero_grad()
                label, prob, curr_action, pred_action_durations, _ = anci(S, R, wl)
                
                loss += loss_fn(curr_action, wl)
                for i in range(prob.shape[0]):
                    loss += loss_mse(pred_action_durations[i][:len(dl[i])-1], dl[i][:-1])
                    if len(fl[i]) > 0:
                        loss += loss_fn(prob[i][:len(fl[i])], fl[i])
                loss.backward()
                
                optimizer1.step()
                total_loss.append(loss)

                label = torch.stack(label).cpu().numpy().T #batch_size*max_len
                for i in range(len(fl)):
                    gt_frame = [int(wl[i])] * int(dl[i][0] * dl[i][-1])
                    pred_frame = [int(torch.argmax(curr_action[i]))] * int(pred_action_durations[i][0] * dl[i][-1])
                    for j in range(1, len(dl[i])-1):
                        gt_frame.extend([int(fl[i][j-1])] * int(dl[i][j] * dl[i][-1]))
                        pred_frame.extend([int(label[i][j-1])] * int(pred_action_durations[i][j] * dl[i][-1]))
                        
                    min_len = min(len(gt_frame), len(pred_frame))
                    if min_len > 0:
                        n += 1
                        total_acc += accuracy_score(gt_frame[:min_len], pred_frame[:min_len])
             
            acc = total_acc/n if n > 0 else 0     
            scheduler1.step()
            print('step1 epoch %d: average loss is %.4f, total time %.2s seconds, acc %.4f' % (e+1, sum(total_loss)/int(full/2), time.time()-s, acc))
                
        #step2: train primary model and self-correction model using full set with ancillary model fixed
        corr = selfcorrModel(args, alpha).to(device)
        params = [{'params':prim.parameters()}, {'params':corr.parameters()}]
        optimizer2 = optim.Adam(params, lr=0.001, betas=(0.99, 0.9999))
        scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[3])
        anci.eval()
        prim.train()
        corr.train()
        print('-------Start training primary model and self-correction model-------')

        for e in range(20):
            s = time.time()
            fullset = DataLoader(dataset=DatasetLoader(args, datapath, 'full'), batch_size=args.batch, shuffle='True', 
                                 collate_fn=collate_fn)
            total_loss = []
            total_acc = 0
            n = 0
            
            for S, R, fl, wl, dl in fullset:
                loss = 0
                optimizer2.zero_grad()
                label, prob_p, curr_action, pred_action_durations_p, attention_p = prim(S, R)
                with torch.no_grad():
                    _, prob_a, _, pred_action_durations_a, attention_a = anci(S, R, wl)
                sfl, sfad = corr(prob_p, prob_a, pred_action_durations_p, pred_action_durations_a)
                
                loss += loss_fn(curr_action, wl)
                
                for i in range(args.max_len):
                    loss += torch.norm(attention_a[i]-attention_p[i], p=2)
                for i in range(prob_p.shape[0]):
                    loss += loss_mse(pred_action_durations_p[i][:len(dl[i])-1], dl[i][:-1])
                    loss += loss_mse(sfad[i][:len(dl[i])-1], dl[i][:-1])
                    if len(fl[i])>0:
                        loss += loss_fn(prob_p[i][:len(fl[i])], fl[i])
                        loss += loss_fn(sfl[i][:len(fl[i])], fl[i])
                loss.backward()
                
                optimizer2.step()
                total_loss.append(loss)

                label = torch.stack(label).cpu().numpy().T #batch_size*max_len
                for i in range(len(fl)):
                    gt_frame = [int(wl[i])] * int(dl[i][0] * dl[i][-1])
                    pred_frame = [int(torch.argmax(curr_action[i]))] * int(pred_action_durations_p[i][0] * dl[i][-1])
                    for j in range(1, len(dl[i])-1):
                        gt_frame.extend([int(fl[i][j-1])] * int(dl[i][j] * dl[i][-1]))
                        pred_frame.extend([int(label[i][j-1])] * int(pred_action_durations_p[i][j] * dl[i][-1]))
                        
                    min_len = min(len(gt_frame), len(pred_frame))
                    if min_len > 0:
                        n += 1
                        total_acc += accuracy_score(gt_frame[:min_len], pred_frame[:min_len])
             
            acc = total_acc/n if n > 0 else 0         
            scheduler2.step()
            print('step2 epoch %d: average loss is %.4f, total time %.2s seconds, acc %.4f' % (e+1, sum(total_loss)/full, time.time()-s, acc))

        #step3: fine-tune primary model using full set and weak set and self-correction model using full set with ancillary model fixed
        print('-------Start fine-tuning primary model and self-correction model-------')
        
        for e in range(20):
            s = time.time()
            fullset = DataLoader(dataset=DatasetLoader(args, datapath, 'full'), batch_size=args.batch, shuffle='True', 
                                 collate_fn=collate_fn)
            weakset = DataLoader(dataset=DatasetLoader(args, datapath, 'weak'), batch_size=args.batch*2, shuffle='True', 
                                 collate_fn=collate_fn)
            total_loss = []
            total_acc = 0
            n = 0
            
            for S, R, fl, wl, dl in fullset:
                loss = 0
                optimizer2.zero_grad()
                label, prob_p, curr_action, pred_action_durations_p, attention_p = prim(S, R)
                with torch.no_grad():
                    _, prob_a, _, pred_action_durations_a, attention_a = anci(S, R, wl)
                sfl, sfad = corr(prob_p, prob_a, pred_action_durations_p, pred_action_durations_a)
                
                loss += loss_fn(curr_action, wl)
                for i in range(args.max_len):
                    loss += torch.norm(attention_a[i]-attention_p[i], p=2)
                for i in range(prob_p.shape[0]):
                    loss += loss_mse(pred_action_durations_p[i][:len(dl[i])-1], dl[i][:-1])
                    loss += loss_mse(sfad[i][:len(dl[i])-1], dl[i][:-1])
                    if len(fl[i])>0:
                        loss += loss_fn(prob_p[i][:len(fl[i])], fl[i])
                        loss += loss_fn(sfl[i][:len(fl[i])], fl[i])
                loss.backward()
                optimizer2.step()
                total_loss.append(loss)

                label = torch.stack(label).cpu().numpy().T #batch_size*max_len
                for i in range(len(fl)):
                    gt_frame = [int(wl[i])] * int(dl[i][0] * dl[i][-1])
                    pred_frame = [int(torch.argmax(curr_action[i]))] * int(pred_action_durations_p[i][0] * dl[i][-1])
                    for j in range(1, len(dl[i])-1):
                        gt_frame.extend([int(fl[i][j-1])] * int(dl[i][j] * dl[i][-1]))
                        pred_frame.extend([int(label[i][j-1])] * int(pred_action_durations_p[i][j] * dl[i][-1]))
                        
                    min_len = min(len(gt_frame), len(pred_frame))
                    if min_len > 0:
                        n += 1
                        total_acc += accuracy_score(gt_frame[:min_len], pred_frame[:min_len])
             
            acc = total_acc/n if n > 0 else 0
                
            for S, R, _, wl, _ in weakset:
                loss = 0
                optimizer2.zero_grad()
                
                _, prob_p, curr_action, pred_action_durations_p, attention_p = prim(S, R)
                with torch.no_grad():
                    _, prob_a, _, pred_action_durations_a, attention_a = anci(S, R, wl)
                    sfl, sfad = corr(prob_p, prob_a, pred_action_durations_p, pred_action_durations_a)
                    
                loss += loss_fn(curr_action, wl)
                loss += loss_mse(pred_action_durations_p, sfad)
                for i in range(args.max_len):
                    loss += torch.norm(attention_a[i]-attention_p[i], p=2)
                for i in range(prob_p.shape[0]):
                    loss += mycrossentropy(prob_p[i], sfl[i])
                loss.backward()
                optimizer2.step()
                total_loss.append(loss)
                      
            print('step3 epoch %d: average loss is %.4f, total time %.2s seconds, acc %.4f' % (e+1, sum(total_loss)/total, time.time()-s, acc)) 
            
        #step4: test
        prim.eval()
        with torch.no_grad():
            testset = DataLoader(dataset=DatasetLoader(args, datapath, 'test'), batch_size=4, shuffle='False', 
                                 collate_fn=collate_fn)
            total_acc = 0
            n = 0
            
            for S, R, fl, wl, dl, ids in testset:
                label, _, curr_action, pred_action_durations, _ = prim(S, R)
                label = torch.stack(label).cpu().numpy().T #batch_size*max_len
                for i in range(len(fl)):
                    gt_frame = [int(wl[i])] * int(dl[i][0] * dl[i][-1])
                    pred_frame = [int(torch.argmax(curr_action[i]))] * int(pred_action_durations[i][0] * dl[i][-1])
                    for j in range(1, len(dl[i])-1):
                        gt_frame.extend([int(fl[i][j-1])] * int(dl[i][j] * dl[i][-1]))
                        pred_frame.extend([int(label[i][j-1])] * int(pred_action_durations[i][j] * dl[i][-1]))
                        
                    min_len = min(len(gt_frame), len(pred_frame))
                    if min_len > 0:
                        n += 1
                        total_acc += accuracy_score(gt_frame[:min_len], pred_frame[:min_len])
             
            acc = total_acc/n if n > 0 else 0
            print('frame-wise accuracy on test set is %.4f' % acc)
            
    torch.save(anci, os.path.join(modelpath, 'fullset_%d_%s_%s_pred-%f_%s_%s_anci' % (full, args.feature_type, args.observation, args.prediction, light, args.self_correction_method)))
    torch.save(prim, os.path.join(modelpath, 'fullset_%d_%s_%s_pred-%f_%s_%s_prim' % (full, args.feature_type, args.observation, args.prediction, light, args.self_correction_method)))
    if args.self_correction_method != 'no':
        torch.save(corr, os.path.join(modelpath, 'fullset_%d_%s_%s_pred-%f_%s_%s_corr' % (full, args.feature_type, args.observation, args.prediction, light, args.self_correction_method)))
    print('Done!')
    
if __name__ == "__main__":
    main()
