# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
import torch.nn.functional as F
class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        ### self.expert_input_dim, self.num_experts#
        self.mmoe_gate = nn.Sequential(torch.nn.Linear(768, 6), nn.Softmax(dim=1))
        # self.tower = MultiLayerPerceptron(self.expert_layer_dims[-1], tower_layer_dims, output_dim=num_classes[i])
        #################
        # unfreeze_layers = ['layer.8', 'layer.9', 'layer.10', 'layer.11'] # 0-11 以及注视掉########code######
        unfreeze_layers = ['layer.8', 'layer.2', 'layer.11']
        for name ,param in self.encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break    
        
        ###################
    def forward(self, code_inputs=None, nl_inputs=None, code_prefix=None): 
        if code_inputs is not None:
            # print("1:", code_inputs.shape)
            output = self.encoder(code_inputs,attention_mask=code_inputs.ne(1), code_prefix=code_prefix)
            outputs = output[0]
            # print("2:", outputs.shape)
            outputs1 = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            # print("3:", torch.nn.functional.normalize(outputs, p=2, dim=1).shape)
            return torch.nn.functional.normalize(outputs1, p=2, dim=1), output[2]
        else:
            output = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))
            outputs = output[0]
            outputs1 = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs1, p=2, dim=1), output[2]

class CLFModel(nn.Module):
    def __init__(self, encoder):
        super(CLFModel, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(768, 6)

        unfreeze_layers = ['layer.11']
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
        

    def forward(self, input_ids=None):
    # def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_hidden_states=True, **kwargs):

        output = self.encoder(input_ids,attention_mask=input_ids.ne(1))

        hidden_state = output[2]
        output = output[0]
         
        # [2, 256, 768]
        # print("2:", outputs.shape)
        outputs = (output*input_ids.ne(1)[:,:,None]).sum(1)/input_ids.ne(1).sum(-1)[:,None]
        # [2, 768]
        output_x = torch.nn.functional.normalize(outputs, p=2, dim=1)
        # print("3:", output_x.shape)
        output_xx = self.linear(output_x)
        # print("4:", output_xx.shape)
        prob=torch.nn.functional.log_softmax(output_xx,-1)
        #rint("5:", prob.shape)
        # return output_x
        return prob, hidden_state

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_size_all):
        super(MLP, self).__init__()
        self.fc1_1 = nn.Linear(input_size, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, output_size)  

        self.fc2_1 = nn.Linear(input_size, hidden_size)  
        self.fc2_2 = nn.Linear(hidden_size, output_size)  

        self.fc3_1 = nn.Linear(input_size, hidden_size)  
        self.fc3_2 = nn.Linear(hidden_size, output_size)

        self.fc4_1 = nn.Linear(input_size, hidden_size)  
        self.fc4_2 = nn.Linear(hidden_size, output_size)

        self.fc5_1 = nn.Linear(input_size, hidden_size)  
        self.fc5_2 = nn.Linear(hidden_size, output_size)
        
        self.fc6_1 = nn.Linear(input_size, hidden_size)  
        self.fc6_2 = nn.Linear(hidden_size, output_size)
        
        self.fc_all = nn.Linear(output_size*6, output_size_all)


    def forward(self, x):

        list_vec = []
        for i in range(6):
            c = x
            c1 = F.relu(getattr(self, f"fc{i+1}_1")(c))  # 修改这里
            c2 = getattr(self, f"fc{i+1}_2")(c1)  # 修改这里
            list_vec.append(c2)

        c_all = torch.cat((list_vec[0], list_vec[1], list_vec[2], list_vec[3], list_vec[4], list_vec[5]), dim=1)
        
        ans = self.fc_all(c_all)
        return ans