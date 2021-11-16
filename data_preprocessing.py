import os.path
import pickle
import numpy as np
import torch

def read_mapping_dict(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict

def get_label_bounds(data_labels):
    labels_uniq = []
    labels_uniq_loc = []
    for kki in range(len(data_labels)):
        sequence_labels, sequence_durations = get_label_length_seq(data_labels[kki])
        labels_uniq.append(sequence_labels)
        labels_uniq_loc.append(sequence_durations)
    return labels_uniq, labels_uniq_loc

def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    length_seq.append(0)
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content))

    return label_seq, length_seq

class DataClass:
    def __init__(self, args, path, mode='full', half=False):
        self.path = path
        self.GT_folder = os.path.join(self.path, 'groundTruth/')
        self.mapping = os.path.join(self.path, 'mapping.txt')
        self.full_split = os.path.join(self.path, 'split/full.split3.bundle')
        self.weak_split = os.path.join(self.path, 'split/weak.split3.bundle')
        self.test_split = os.path.join(self.path, 'split/test.split.bundle')
        
        self.obs = float(args.observation[-3:]) #observation portion
        self.pred = args.prediction #prediction portion
        self.fps = args.fps #video's fps
        
        self.curr_label = dict()
        self.future_labels = dict()
        self.future_durations = dict()
        
        actions_dict = read_mapping_dict(self.mapping)
        if args.feature_type == 'gt' or args.feature_type == 'fisher' or args.feature_type == 'I3D':
            if mode == 'full':
                self.data_feat, data_labels = self.load_data_features(args, self.full_split, actions_dict, half)
            elif mode == 'weak':
                self.data_feat, data_labels = self.load_data_features(args, self.weak_split, actions_dict, half)
            else:
                self.data_feat, data_labels = self.load_data_features(args, self.test_split, actions_dict, half)
            
        elif args.feature_type == 'fisher_label' or args.feature_type == 'I3D_label':
            if mode == 'full':
                self.data_feat, data_labels = self.load_seg_outs(args, self.full_split, actions_dict, mode, half)
            elif mode == 'weak':
                self.data_feat, data_labels = self.load_seg_outs(args, self.weak_split, actions_dict, mode, half)
            else:
                self.data_feat, data_labels = self.load_seg_outs(args, self.test_split, actions_dict, mode, half)
                
        else:
            if mode == 'full':
                self.data_feat, data_labels = self.load_seg_outs_concat(args, self.full_split, actions_dict, mode, half)
            elif mode == 'weak':
                self.data_feat, data_labels = self.load_seg_outs_concat(args, self.weak_split, actions_dict, mode, half)
            else:
                self.data_feat, data_labels = self.load_seg_outs_concat(args, self.test_split, actions_dict, mode, half)
                
        labels_uniq, labels_uniq_loc = get_label_bounds(data_labels)
        
        counter_index = 0
        for kki in range(0, len(data_labels)):
            mi_labels = data_labels[kki] #a video's frame-wise label
            video_len = len(mi_labels)
            sequence_labels = labels_uniq[kki]
            sequence_durations = labels_uniq_loc[kki]
            
            current_stop = int(len(mi_labels) * self.obs) #the last frame of the observation part
            pred_stop = int(len(mi_labels) * (self.obs + self.pred)) #the last frame of the prediction part
            stop_index = 0
            for ioi in range(len(sequence_durations) - 1):
                if sequence_durations[ioi] <= current_stop:
                    stop_index = ioi
            #the order of the last action in the observation part
            
            list_future_labels = []
            list_future_durations = [min(pred_stop, sequence_durations[stop_index+1]) - current_stop] #current action duration (within the prediction part)
            val_curr_label = sequence_labels[stop_index]
            if stop_index + 1 != len(sequence_labels):
                for izi in range(stop_index + 1, len(sequence_labels)):
                    if sequence_durations[izi] <= pred_stop:
                        list_future_durations.append(min(pred_stop - sequence_durations[izi], sequence_durations[izi+1] - sequence_durations[izi]))
                        list_future_labels.append(sequence_labels[izi])
            
            self.curr_label[str(counter_index)] = torch.tensor(val_curr_label).long() #current action
            self.future_labels[str(counter_index)] = torch.Tensor(list_future_labels).long() #future actions
            self.future_durations[str(counter_index)] = torch.cat((torch.Tensor(list_future_durations)/video_len, torch.Tensor([video_len]))) #future actions durations
            counter_index = counter_index + 1
                
    def load_data_features(self, args, split_load, actions_dict, half=False):  
        file_ptr = open(split_load, 'r')
        if half:
            content_all = file_ptr.read().split('\n')[:-1]
            content_all = content_all[:int(len(content_all)/2)]
        else:    
            content_all = file_ptr.read().split('\n')[:-1]
        if args.dataset == 'breakfast':
            content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]
    
        data_all = []
        label_all = []
        for content in content_all:
            file_ptr = open(self.GT_folder + content, 'r')
            curr_gt = file_ptr.read().split('\n')[:-1]
    
            curr_data = []
            if args.feature_type == 'fisher':
                if args.dataset == 'breakfast':
                    loc_curr_data = self.path + 'fisher/' + os.path.splitext(content)[0] + '.txt'
                    curr_data = np.loadtxt(loc_curr_data, dtype='float32')
                    curr_data = curr_data[:, 1:65] #n*64 (n is the number of frame)
                else: #args.dataset == 'salad'
                    loc_curr_data = self.path + 'fisher/' + os.path.splitext(content)[0] + '-New.txt'
                    curr_data = np.loadtxt(loc_curr_data, dtype='float32')
                    curr_data = curr_data[:, 1:65] #n*64
            elif args.feature_type == 'I3D':
                if args.dataset == 'breakfast':
                    loc_curr_data = self.path + 'I3D/' + os.path.splitext(content)[0]
                    curr_data = np.loadtxt(loc_curr_data, dtype='float32') #n*400
                else: #args.dataset == 'salad'
                    loc_curr_data = self.path + 'I3D/' + os.path.splitext(content)[0] + '.npy'
                    curr_data = np.load(loc_curr_data).T #n*2048
            else: #args.feature_type == 'gt'
                for iik in range(len(curr_gt)):
                    ind_label = actions_dict[curr_gt[iik]]
                    curr_data_vec = np.zeros(args.n_classes)
                    curr_data_vec[ind_label] = 1.0
                    curr_data.append(curr_data_vec)
                curr_data = np.array(curr_data) #n*n_classes(one-hot)
    
            label_curr_video = []
            for iik in range(len(curr_gt)):
                label_curr_video.append(actions_dict[curr_gt[iik]])
    
            data_all.append(torch.tensor(curr_data, dtype=torch.float32))
            label_all.append(label_curr_video)
        return data_all, label_all
    
    def load_seg_outs(self, args, split_load, actions_dict, mode, half=False):
        file_ptr = open(split_load, 'r')
        if half:
            content_all = file_ptr.read().split('\n')[:-1]
            content_all = content_all[:int(len(content_all)/2)]
        else:    
            content_all = file_ptr.read().split('\n')[:-1]
        content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]
            
        data_all = []
        label_all = []
        
        if mode == 'full' or mode == 'weak':
            for content in content_all:
                file_ptr = open(self.GT_folder + content, 'r')
                curr_gt = file_ptr.read().split('\n')[:-1]
                
                label_curr_video = []
                for iik in range(len(curr_gt)):
                    label_curr_video.append(actions_dict[curr_gt[iik]])
                    
                curr_data = []
                for iik in range(len(label_curr_video)):
                    ind_label = label_curr_video[iik]
                    curr_data_vec = np.zeros(args.n_classes)
                    curr_data_vec[ind_label] = 1.0
                    curr_data.append(curr_data_vec)
                    
                curr_data = np.array(curr_data)
                data_all.append(torch.tensor(curr_data, dtype=torch.float32))
                label_all.append(label_curr_video)
                
        else:
            if args.feature_type == 'fisher_label':
                # current split for fisher vector based segmentation labels
                segmentation_location = os.path.join(self.path, 'seg_fisher')
                
                for content in content_all:
                    file_ptr = open(self.GT_folder + content, 'r')
                    curr_gt = file_ptr.read().split('\n')[:-1]
                    
                    label_curr_video = []
                    for iik in range(len(curr_gt)):
                        label_curr_video.append(actions_dict[curr_gt[iik]])
                        
                    # read fisher based segmentation labels
                    file_ptr_fisher = open(segmentation_location + '/split1/' + args.observation + '/' + content, 'r')
                    fisher_seg_labels = file_ptr_fisher.read().split('\n')[:-1]
                    
                    curr_data = []
                    for iik in range(len(fisher_seg_labels)):
                        ind_label = actions_dict[fisher_seg_labels[iik]]
                        curr_data_vec = np.zeros(args.n_classes)
                        curr_data_vec[ind_label] = 1.0
                        curr_data.append(curr_data_vec)
                    
                    curr_data = np.array(curr_data)
                    data_all.append(torch.tensor(curr_data, dtype=torch.float32))
                    label_all.append(label_curr_video)
                
            else:
                counter = 0
                # read segmentation labels based on i3d features
                file_name = os.path.join(self.path, 'seg_I3D') + '/' + 'seg_ours_2_split1.pickle'
                with open(file_name, 'rb') as handle:
                    segmentation_data = pickle.load(handle)
                    
                for content in content_all:
                    file_ptr = open(self.GT_folder + content, 'r')
                    curr_gt = file_ptr.read().split('\n')[:-1]
                    
                    label_curr_video = []
                    for iik in range(len(curr_gt)):
                        label_curr_video.append(actions_dict[curr_gt[iik]])
                        
                    # read i3d based segmentation labels
                    i3d_seg_labels = segmentation_data[counter]
                    counter = counter + 1
                    
                    curr_data = []
                    for iik in range(len(i3d_seg_labels)):
                        ind_label = i3d_seg_labels[iik]
                        curr_data_vec = np.zeros(args.n_classes)
                        curr_data_vec[ind_label] = 1.0
                        curr_data.append(curr_data_vec)
                        
                    curr_data = np.array(curr_data)
                    data_all.append(torch.tensor(curr_data, dtype=torch.float32))
                    label_all.append(label_curr_video)
                    
        return data_all, label_all
    
    def load_seg_outs_concat(self, args, split_load, actions_dict, mode, half=False):
        file_ptr = open(split_load, 'r')
        if half:
            content_all = file_ptr.read().split('\n')[:-1]
            content_all = content_all[:int(len(content_all)/2)]
        else:    
            content_all = file_ptr.read().split('\n')[:-1]
        content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]

        data_all = []
        label_all = []
        
        if args.feature_type == 'fisher_label_cat':
            # current split for fisher vector based segmentation labels
            segmentation_location = os.path.join(self.path, 'seg_fisher')
            
            for content in content_all:
                #fisher feature
                loc_curr_data = self.path+'fisher/' + os.path.splitext(content)[0] + '.txt'
                curr_data = np.loadtxt(loc_curr_data, dtype='float32')
                curr_data = curr_data[:, 1:65] #n*64(n指帧数)
                
                #gt label
                file_ptr = open(self.GT_folder + content, 'r')
                curr_gt = file_ptr.read().split('\n')[:-1]
                label_curr_video = []
                for iik in range(len(curr_gt)):
                    label_curr_video.append(actions_dict[curr_gt[iik]])
                    
                #one-hot feature
                curr_data_feat = []
                if mode == 'full' or mode == 'weak':
                    #gt one-hot label
                    for iik in range(len(curr_gt)):
                        curr_data_vec = np.zeros(args.n_classes)
                        curr_data_vec[actions_dict[curr_gt[iik]]] = 1.0
                        curr_data_feat.append(curr_data_vec)
    
                else:
                    # read fisher based segmentation labels
                    file_ptr_fisher = open(segmentation_location + '/split1/' + args.observation + '/' + content, 'r')
                    fisher_seg_labels = file_ptr_fisher.read().split('\n')[:-1]
                    for iik in range(len(fisher_seg_labels)):
                        curr_data_vec = np.zeros(args.n_classes)
                        curr_data_vec[actions_dict[fisher_seg_labels[iik]]] = 1.0
                        curr_data_feat.append(curr_data_vec)

                curr_data_feat = np.array(curr_data_feat) #n*n_classes
                minlen = min(len(curr_data_feat), len(curr_data))
                curr_data = np.concatenate((curr_data_feat[:minlen], curr_data[:minlen]), axis=1)
                data_all.append(torch.tensor(curr_data, dtype=torch.float32))
                label_all.append(label_curr_video)
                
        else:
            # read segmentation labels based on i3d features
            file_name = os.path.join(self.path, 'seg_I3D') + '/' + 'seg_ours_2_split1.pickle'
            with open(file_name, 'rb') as handle:
                segmentation_data = pickle.load(handle)
                
            counter = 0
            for content in content_all:
                #I3D feature
                loc_curr_data = self.path + 'I3D/' + os.path.splitext(content)[0]
                curr_data = np.loadtxt(loc_curr_data, dtype='float32') #n*400
                
                #gt label
                file_ptr = open(self.GT_folder + content, 'r')
                curr_gt = file_ptr.read().split('\n')[:-1]
                label_curr_video = []
                for iik in range(len(curr_gt)):
                    label_curr_video.append(actions_dict[curr_gt[iik]])
                
                #one-hot label
                curr_data_feat = []
                if mode == 'full' or mode == 'weak':
                    #gt one-hot label
                    for iik in range(len(curr_gt)):
                        curr_data_vec = np.zeros(args.n_classes)
                        curr_data_vec[actions_dict[curr_gt[iik]]] = 1.0
                        curr_data_feat.append(curr_data_vec)
                    
                else:
                    # read i3d based segmentation labels
                    i3d_seg_labels = segmentation_data[counter]
                    counter = counter + 1
                    for iik in range(len(i3d_seg_labels)):
                        curr_data_vec = np.zeros(args.n_classes)
                        curr_data_vec[i3d_seg_labels[iik]] = 1.0
                        curr_data_feat.append(curr_data_vec)
                        
                curr_data_feat = np.array(curr_data_feat) #n*n_classes
                minlen = min(len(curr_data_feat), len(curr_data))
                curr_data = np.concatenate((curr_data_feat[:minlen], curr_data[:minlen]), axis=1)
                data_all.append(torch.tensor(curr_data, dtype=torch.float32))
                label_all.append(label_curr_video)

        return data_all, label_all
