'''
input: a video
output: predicted frame-wise action
Primary prediction model generates a frame-wise prediction of actions given an video. 
This is the main model that is subject to the training and is used at test time.
'''
import torch.nn as nn
from blocks import TABlock
import torch
import torch.nn.functional as F

class primModel(nn.Module):
    def __init__(self, args, video_feat_dim):
        super(primModel, self).__init__()

        self.n_classes = args.n_classes
        self.hidden_size = args.hidden_dim_LSTM
        self.num_TAB = len(args.startpoints_R)
        self.linear_dim = args.linear_dim
        self.max_len = args.max_len
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        self.TABs = nn.ModuleList([TABlock(args, video_feat_dim) for _ in range(self.num_TAB)])
        self.cls_layer = nn.ModuleList([nn.Sequential(nn.Linear(in_features=2*self.linear_dim,out_features=self.n_classes), nn.Softmax(dim=1)) for _ in range(self.num_TAB)])
        self.cls_curr_duration = nn.Linear(in_features=self.num_TAB*self.linear_dim, out_features=1)
        self.lstm_linear = nn.Linear(in_features=(2*self.num_TAB)*self.linear_dim + self.num_TAB*self.n_classes, out_features=self.linear_dim+1)
        self.lstm = nn.LSTM(self.linear_dim+1, self.hidden_size, batch_first=True)
        
        self.pred_class = nn.Linear(in_features=self.hidden_size, out_features=self.n_classes)
        self.pred_duration = nn.Linear(in_features=self.hidden_size+self.linear_dim, out_features=1)
        self.embed = nn.Embedding(self.n_classes, self.linear_dim)
        self.attn = nn.Linear(in_features=self.hidden_size, out_features=self.linear_dim, bias=False)
        
    def forward(self, S_list, R_list):
        S_ppps = []
        R_ppps = []
        Y = []
        
        for i in range(len(R_list)):
            S_ppp, R_ppp = self.TABs[i](S_list, R_list[i])
            S_ppps.append(S_ppp)
            R_ppps.append(R_ppp)
            Y.append(self.cls_layer[i](torch.cat((S_ppp, R_ppp), 1)))       
            
        lstm_input = torch.cat((Y+S_ppps+R_ppps), 1) #batch_size*(2*num_TAB*linear_dim+num_TAB*n_classes)
        lstm_input = self.lstm_linear(lstm_input).unsqueeze(1) #batch_size*1*(linear_dim+1)
        
        curr_action_duration = self.cls_curr_duration(torch.cat(R_ppps, 1)) #batch_size*1
        pred_action_durations = [curr_action_duration]
        
        batch_size = lstm_input.size(0)
        pred_class_labels = []
        pred_class_probs = []
        attentions = []
        states = None
        prev_hiddens = torch.zeros(batch_size, self.hidden_size).to(self.device)
        
        for i in range(self.max_len):
            hiddens, states = self.lstm(lstm_input, states)
            hiddens = hiddens.squeeze(1) #batch_size*hidden_size
            outputs = self.pred_class(hiddens)
            
            attention = F.softmax(torch.matmul(self.attn(hiddens).unsqueeze(1)/(self.linear_dim ** 0.5), torch.stack(S_ppps, 1).permute(0,2,1)), dim=-1) #batch_size*1*3
            attention = torch.matmul(attention, torch.stack(S_ppps, 1)).view(batch_size, -1) #batch_size*linear_dim
            attentions.append(attention)
            duration = self.pred_duration(torch.cat((attention, prev_hiddens), 1)) #batch_size*1
            
            predicted_class = outputs.max(1)[1] #batch_size
            pred_class_prob = F.softmax(outputs, 1) #batch_size*n_classes
            
            pred_class_labels.append(predicted_class)
            pred_class_probs.append(pred_class_prob)
            pred_action_durations.append(duration)
            lstm_input = torch.cat((self.embed(predicted_class), duration), 1).unsqueeze(1) #batch_size*1*(linear_dim+1)
            prev_hiddens = hiddens
            
        curr_action = torch.sum(torch.stack(Y, 0), 0) #current action: batch_size*n_classes
        pred_class_probs = torch.stack(pred_class_probs, 1) #batch_size*max_len*n_classes
        pred_action_durations = torch.cat(pred_action_durations, 1) #batch_size*(max_len+1)
        return pred_class_labels, pred_class_probs, curr_action, pred_action_durations, attentions