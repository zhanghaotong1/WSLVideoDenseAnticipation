import torch
import torch.utils.data as data
from data_preprocessing import DataClass

class DatasetLoader(data.Dataset):
    def __init__(self, args, path, mode, half=False):
        self.dataset = DataClass(args, path, mode, half)
        self.obs = float(args.observation[-3:]) #observation portion
        self.pred = args.prediction #prediction portion
        self.fps = args.fps
        self.len_R = args.len_R
        self.startpoints_R = args.startpoints_R
        self.len_S_list = args.len_S_list
        self.args = args
        self.mode = mode

        self.features = self.dataset.data_feat #list, one element is the feature of one video (tensor) 
        self.curr_label = self.dataset.curr_label #dict, key is video index, value is its current action label
        self.future_labels = self.dataset.future_labels #dict, key is video index, value is its future labels list, could be an empty list
        self.future_durations = self.dataset.future_durations #dict, key is video index, value is its current and future action duration (within the prediction part)
        
    def cut(self, feature, curr_label, future_labels, durations):
        '''
        feature : tensor (n*dim)
            feature of a video, n is the number of frames, dim is the dimension of each frame.
        curr_label: torch.longtensor, label of current action
        future_labels : torch.longtensor, zero or several labels
        Return S_list, R_list, groundtruth label for predict part and weak label
        '''
        if (self.args.feature_type == 'fisher_label' or self.args.feature_type == 'I3D_label' or self.args.feature_type == 'fisher_label_cat' or self.args.feature_type == 'I3D_label_cat') and self.mode == 'test':
            obs = feature
        else:
            obs = feature[:int(len(feature) * self.obs), :] #first obs (0.2 or 0.3) portion of videos as observation part
        full_label = future_labels #ground truth for prediction part
        weak_label = curr_label #weak label: current action label
        durations = durations
        
        recent_snippets = [] #R_list
        spanning_snippets = [] #S_list
        for scale in self.len_S_list:
            curr = []
            a = len(obs)/scale
            for i in range(scale):
                curr.append(torch.max(obs[int(i*a):int((i+1)*a)], 0)[0].squeeze())
            spanning_snippets.append(torch.stack(curr))
            
        for sp in self.startpoints_R:
            curr = []
            recent = obs[int(max(0, len(obs)-sp*self.fps)):, :]
            a = len(recent)/self.len_R
            for i in range(self.len_R):
                curr.append(torch.max(recent[int(i*a):int((i+1)*a)], 0)[0].squeeze())
            recent_snippets.append(torch.stack(curr))
            
        return (spanning_snippets, recent_snippets, full_label, weak_label, durations)
        
    def __getitem__(self, index):
        return self.cut(self.features[index], self.curr_label[str(index)], self.future_labels[str(index)], self.future_durations[str(index)]) #a tuple
    
    def __len__(self):
        return len(self.features)
    
def collate_fn(data):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    S_list = [[] for _ in range(3)] #3 is the length of len_S_list
    R_list = [[] for _ in range(3)] #3 is the length of startpoints_R
    fl = []
    wl = []
    dl = []
    
    for d in data:
        curr_s = d[0] #List: len_S_i*feat_dim
        curr_r = d[1] #List: len_R*feat_dim
        for i in range(len(curr_s)):
            S_list[i].append(curr_s[i])
        for i in range(len(curr_r)):
            R_list[i].append(curr_r[i])
        fl.append(d[2].to(device)) #List: each element is a tensor of future action labels
        wl.append(d[3])
        dl.append(d[4].to(device)) #List: each element is a tensor of current and future action durations
        
    S_list = [torch.stack(s).to(device) for s in S_list] #List: each element is batch*len_S_i*feat_dim
    R_list = [torch.stack(r).to(device) for r in R_list] #List: each element is batch*len_R*feat_dim
    wl = torch.stack(wl, 0).to(device) #batch
    return S_list, R_list, fl, wl, dl