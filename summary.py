import torch
from torchinfo import summary

model = torch.load('saves/DocJERE-DocRED-roberta-base-e50-bs4_lr2e-5_maxspan15_md1-reduceLRpat2_cr1_et1_re1_test-similarity-bestf1-regularConv2D_2024-08-25_09-32-56.pt')
#model.cuda()
print(model)