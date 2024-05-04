import torch
import torch.nn as nn
from copy import deepcopy

class SpaceDecouplingHead(nn.Module):
    def __init__(self, num_tasks, num_classes):
        super(SpaceDecouplingHead, self).__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.linear = nn.Linear(num_classes, num_tasks * num_classes)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.num_tasks, self.num_classes)
        return x


class LLL_Net(nn.Module):
    def __init__(self, model, num_tasks, remove_existing_head=False, use_space_decoupling=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        self.num_tasks = num_tasks
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off the last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts the last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = torch.tensor([]) 
        self.task_offset = torch.tensor([0])
        self._initialize_weights()

        if use_space_decoupling:
            # Add Space Decoupling head
            self.space_decoupling_head = SpaceDecouplingHead(num_tasks, self.out_size)

    def add_head(self, num_outputs):
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])
        
    def forward(self, x, return_features=False):
        x = self.model(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        pass
