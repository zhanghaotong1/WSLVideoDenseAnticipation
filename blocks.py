import torch
import torch.nn as nn
import torch.nn.functional as F

class NONLocalBlock(nn.Module):
    #Non Local Block
    def __init__(self, args, dim_1, dim_2, video_feat_dim):
        super(NONLocalBlock, self).__init__()

        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.video_feat_dim = video_feat_dim
        self.latent_dim = args.conv_dim_NLB
        self.dropout = args.dropout_NLB
        self.initnn = True# initnn is used in theta, phi, and g.
        self.initnn2 = True# initnn2 is used in the final linear layer.

        self.theta = nn.Conv1d(in_channels=self.dim_2,
                               out_channels=self.latent_dim,
                               kernel_size=1, stride=1, padding=0)
        if self.initnn:
            nn.init.xavier_normal_(self.theta.weight)
            nn.init.constant_(self.theta.bias, 0)

        self.phi = nn.Conv1d(in_channels=self.dim_1,
                             out_channels=self.latent_dim,
                             kernel_size=1, stride=1, padding=0)
        if self.initnn:
            nn.init.xavier_normal_(self.phi.weight)
            nn.init.constant_(self.phi.bias, 0)

        self.g = nn.Conv1d(in_channels=self.dim_1,
                           out_channels=self.latent_dim,
                           kernel_size=1, stride=1, padding=0)
        if self.initnn:
            nn.init.xavier_normal_(self.g.weight)
            nn.init.constant_(self.g.bias, 0)

        self.final_layers = nn.Sequential(
                            nn.LayerNorm(torch.Size([self.latent_dim, self.video_feat_dim])),
                            nn.ReLU(),
                            nn.Conv1d(in_channels=self.latent_dim,
                                      out_channels=self.dim_2,
                                      kernel_size=1, stride=1, padding=0),
                            nn.Dropout(p=self.dropout),
        )
        if self.initnn2:
            nn.init.xavier_normal_(self.final_layers[2].weight)
            nn.init.constant_(self.final_layers[2].bias, 0)


    def forward(self, input1, input2):
        #input1: batch_size*dim_1*video_feat_dim
        #input2: batch_size*dim_2*video_feat_dim
        
        theta_x = self.theta(input2).permute(0, 2, 1) #batch_size*video_feat_dim*latent_dim
        phi_x = self.phi(input1) #batch_size*latent_dim*video_feat_dim
        theta_phi = torch.matmul(theta_x, phi_x) #batch_size*video_feat_dim*video_feat_dim
        p_x = F.softmax(theta_phi, dim=-1) #batch_size*video_feat_dim*video_feat_dim
        
        g_x = self.g(input1).permute(0, 2, 1) #batch_size*video_feat_dim*latent_dim
        t_x = torch.matmul(p_x, g_x).permute(0, 2, 1).contiguous() #batch_size*latent_dim*video_feat_dim
        W_t = self.final_layers(t_x) #batch_size*dim_2*video_feat_dim
        z_x = W_t + input2 #batch_size*dim_2*video_feat_dim
        
        return z_x
    
class CouplingBlock(nn.Module):
    #Coupling Block
    def __init__(self, args, dim_S, dim_R, video_feat_dim):
        super(CouplingBlock, self).__init__()

        self.dropout = args.dropout_CB
        self.video_feat_dim = video_feat_dim
        self.linear_dim = args.linear_dim
        
        self.dim_S = dim_S
        self.dim_R = dim_R
        
        self.coupBlock1 = NONLocalBlock(args, self.dim_S, self.dim_S, video_feat_dim)
        self.coupBlock2 = NONLocalBlock(args, self.dim_S, self.dim_R, video_feat_dim)
        self.final_SR = nn.Sequential(
            nn.Linear(in_features = (self.dim_S+2*self.dim_R)*self.video_feat_dim, out_features = self.linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout))
        self.final_RR = nn.Sequential(
            nn.Linear(in_features = 2*self.dim_R*self.video_feat_dim, out_features = self.linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout))

    def forward(self, S, R):
        #S: batch_size*dim_S*video_feat_dim
        #R: batch_size*dim_R*video_feat_dim
        batch_size = S.size(0)
        
        S_p = F.relu(self.coupBlock1(S, S)) #batch_size*dim_S*video_feat_dim
        R_p = F.relu(self.coupBlock2(S_p, R)) #batch_size*dim_R*video_feat_dim
        
        R_pp = torch.cat((R_p, R), 1).view(batch_size, -1) #batch_size*(2*dim_R*video_feat_dim)
        S_pp = torch.cat((S_p.view(batch_size, -1).contiguous(), R_pp), 1).view(batch_size, -1) #batch_size*(dim_S*video_feat_dim+2*dim_R*video_feat_dim)
        
        S_pp = self.final_SR(S_pp) #batch_size*linear_dim
        R_pp = self.final_RR(R_pp) #batch_size*linear_dim
        return S_pp, R_pp
    
class CouplingBlock_light(nn.Module):
    #Coupling Block
    def __init__(self, args, dim_S, dim_R, video_feat_dim):
        super(CouplingBlock_light, self).__init__()

        self.dropout = args.dropout_CB
        self.video_feat_dim = video_feat_dim
        self.linear_dim = args.linear_dim
        
        self.dim_S = dim_S
        self.dim_R = dim_R

        self.coupBlock = NONLocalBlock(args, self.dim_S, self.dim_R, video_feat_dim)
        self.final_SR = nn.Sequential(
            nn.Linear(in_features = (self.dim_S+2*self.dim_R)*self.video_feat_dim, out_features = self.linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout))
        self.final_RR = nn.Sequential(
            nn.Linear(in_features = 2*self.dim_R*self.video_feat_dim, out_features = self.linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout))

    def forward(self, S, R):
        #S: batch_size*dim_S*video_feat_dim
        #R: batch_size*dim_R*video_feat_dim
        batch_size = S.size(0)
        
        R_p = F.relu(self.coupBlock(S, R)) #batch_size*dim_R*video_feat_dim
        
        R_pp = torch.cat((R_p, R), 1).view(batch_size, -1) #batch_size*(2*dim_R*video_feat_dim)
        S_pp = torch.cat((S.view(batch_size, -1).contiguous(), R_pp), 1).view(batch_size, -1) #batch_size*(dim_S*video_feat_dim+2*dim_R*video_feat_dim)
        
        S_pp = self.final_SR(S_pp) #batch_size*linear_dim
        R_pp = self.final_RR(R_pp) #batch_size*linear_dim
        return S_pp, R_pp

class TABlock(nn.Module):
    #Temporal Aggregation Block
    def __init__(self, args, video_feat_dim):
        super(TABlock, self).__init__()

        self.linear_dim  = args.linear_dim
        self.video_feat_dim = video_feat_dim
        self.len_R = args.len_R
        self.len_S_list = args.len_S_list
        self.S_num = len(self.len_S_list)
        self.dropout = args.dropout_TAB
        self.light = args.light

        if self.light:
            self.CBs = nn.ModuleList([CouplingBlock_light(args, len_S, self.len_R, video_feat_dim) for len_S in self.len_S_list])
        else:
            self.CBs = nn.ModuleList([CouplingBlock(args, len_S, self.len_R, video_feat_dim) for len_S in self.len_S_list])
#        self.final_RRR = nn.Sequential(
#            nn.Linear(in_features = self.S_num*self.linear_dim, out_features = self.linear_dim),
#            nn.ReLU(),
#            nn.Dropout(self.dropout))
        self.final_RRR = nn.Linear(in_features = self.S_num*self.linear_dim, out_features = self.linear_dim)
 
    def forward(self, S_list, R):
        S_pps = []
        R_pps = []
        
        for i in range(len(S_list)):
            S_pp, R_pp = self.CBs[i](S_list[i], R)
            S_pps.append(S_pp)
            R_pps.append(R_pp) 
            
        R_ppp = torch.cat(R_pps, 1) #batch_size*(3*linear_dim)
        R_ppp = self.final_RRR(R_ppp) #batch_size*linear_dim
        S_ppp = torch.stack(S_pps, 0) #3*batch_size*linear_dim
        S_ppp = torch.max(S_ppp, 0)[0].view(-1, self.linear_dim) #batch_size*linear_dim

        return S_ppp, R_ppp
