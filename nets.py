import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(354, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # initialize linear layers
        for m in self.modules():
            if type(m) is nn.Linear:
                torch.nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.model(x)
        return out


class Cnn1c(nn.Module):
    def __init__(self):
        super(Cnn1c, self).__init__()
        self.convLayer1 = nn.Sequential(
            nn.Conv1d(1, 16, 5),                    # 2*(354) -> 16*(350)
            nn.ReLU(),
        )
        self.convLayer2 = nn.Sequential(
            nn.Conv1d(16, 16, 5),                   # 16*(350) -> 16*(346)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 16*(346) -> 16*(173)
        )
        self.convLayer3 = nn.Sequential(
            nn.Conv1d(16, 32, 5),                   # 16*(173) -> 32*(169)
            nn.ReLU(),
        )
        self.convLayer4 = nn.Sequential(
            nn.Conv1d(32, 32, 5),                   # 32*(169) -> 32*(165)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # 32*(55) -> 32*(55)
        )
        self.convLayer5 = nn.Sequential(
            nn.Conv1d(32, 64, 5),                   # 32*(55) -> 64*(51)
            nn.ReLU(),
        )
        self.convLayer6 = nn.Sequential(
            nn.Conv1d(64, 64, 7),  # 64*(51) -> 64*(45)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # 64*(45) -> 64*(15)
        )
        self.linearLayer = nn.Sequential(
            nn.Linear(64*15, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 3),
        )

    def forward(self, x):
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        x = self.convLayer3(x)
        x = self.convLayer4(x)
        x = self.convLayer5(x)
        x = self.convLayer6(x)
        x = x.view(x.size(0), -1)
        x = self.linearLayer(x)
        return x


class Cnn2c(nn.Module):
    def __init__(self):
        super(Cnn2c, self).__init__()
        self.convLayer = nn.Sequential(
            nn.Conv1d(2, 16, 5),                    # 2*(354) -> 16*(350)
            nn.ReLU(),
            nn.Conv1d(16, 16, 5),                   # 16*(350) -> 16*(346)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 16*(346) -> 16*(173)

            nn.Conv1d(16, 32, 5),                   # 16*(173) -> 32*(169)
            nn.ReLU(),
            nn.Conv1d(32, 32, 5),                   # 32*(169) -> 32*(165)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # 32*(55) -> 32*(55)

            nn.Conv1d(32, 64, 5),                   # 32*(55) -> 64*(51)
            nn.ReLU(),
            nn.Conv1d(64, 64, 7),                   # 64*(51) -> 64*(45)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # 64*(45) -> 64*(15)
        )
        self.linearLayer = nn.Sequential(
            nn.Linear(64 * 15, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 3),
        )

    def forward(self, x):
        x = self.convLayer(x)
        x = x.view(x.size(0), -1)
        x = self.linearLayer(x)
        return x