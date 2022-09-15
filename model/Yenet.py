import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

srm_16filters = np.load("./srm_16_filters.npy")
srm_minmax = np.load("./minmax_filters.npy")
srm_filters = np.concatenate((srm_16filters, srm_minmax), axis=0)

srm_filters = torch.from_numpy(srm_filters).to(device=device, dtype=torch.float)
srm_filters = torch.autograd.Variable(srm_filters, requires_grad=True)
# print(srm_filters.shape)


class Yenet(nn.Module):
    def __init__(self):
        super(Yenet, self).__init__()

        self.tlu = nn.Hardtanh(min_val=-3.0, max_val=3.0)

        self.conv2 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=0)

        self.conv3 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=0)

        self.conv4 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=0)

        self.conv5 = nn.Conv2d(30, 32, kernel_size=5, stride=1, padding=0)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)

        self.conv7 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)

        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)

        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=3, padding=0)

        self.fc = nn.Linear(16 * 3 * 3, 2)

    def forward(self, x):
        out = self.tlu(F.conv2d(x, srm_filters))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = F.relu(self.conv5(out))
        out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=0)
        out = F.relu(self.conv6(out))
        out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=0)
        out = F.relu(self.conv7(out))
        out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=0)
        out = F.relu(self.conv8(out))
        out = F.relu(self.conv9(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
