import torch
import pdb

# path = "/home/cmz/cf-vit/checkpoints/cf-deit-s-7x7.pth"
# path = "/home/cmz/cf-vit/checkpoints/cf-deit-s-9x9-81.9.pth"
# path = "/home/cmz/cf-vit/checkpoints/cf-lvvit-s-7x7-83.6.pth"
path = "/home/cmz/cf-vit/checkpoints/cf-lvvit-s-9x9-84.4.pth"
data = torch.load(path)
new_data = {}
# pdb.set_trace()

# data['model'][]
data['model']['cls_token'] = data['model']['cls_token_list.1']
data['model'].pop('cls_token_list.1')
data['model'].pop('cls_token_list.0')
torch.save(data,path)
# new_data['state_dict'] = data['state_dict_ema']
# new_data['model'] = data['state_dict']

# new_data['flops'] = [1.1, 4.0]
# new_data['flops'] = [1.831,6.698]
# new_data['flops'] = [1.574, 6.096]
# new_data['flops'] = [2.617, 10.3]
# torch.save(new_data,path)
# pdb.set_trace()