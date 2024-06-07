import torch
import torch.nn as nn
import torch.nn.functional as F
from model_transformer import SimpleViT, SimpleViT_feature

class M5_fusion_all_transformer(nn.Module):
    def __init__(self, n_input=1, n_output=2, stride=1, n_channel=32):
        super().__init__()

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=7, stride=stride, padding='same', padding_mode="replicate")
        self.bn1 = nn.BatchNorm1d(n_channel)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=5, padding='same', padding_mode="replicate")
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.AvgPool1d(2)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3, padding='same', padding_mode="replicate")
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.AvgPool1d(2)

        self.fcf = nn.Sequential(
            nn.LayerNorm(37),
            nn.Linear(37, 96)
        )
        self.fcf2 = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 64)
        )

        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=1, padding='same', padding_mode="replicate")
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.conv5 = nn.Conv1d(2 * n_channel, n_channel, kernel_size=3, padding='same', padding_mode="replicate")
        self.bn5 = nn.BatchNorm1d(n_channel)
        self.vit = SimpleViT_feature(seq_len=128, patch_size=16, num_classes=n_output, dim=512, depth=6, heads=8, mlp_dim=64, channels=n_channel)

        self.fc1 = nn.Sequential(
            nn.LayerNorm(512 + 64),
            nn.Linear(512 + 64, 128)
        )
        self.fc2 = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, n_output)
        )

        self.dropout = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)

    def forward(self, x, age, bmi, features):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        # x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.conv5(x)
        x = F.relu(self.bn5(x))

        x = self.vit(x)
        
        features = F.relu(self.fcf(torch.concat((age, bmi, features), axis=2)))
        features = self.dropout3(features)
        features = F.relu(self.fcf2(features))
        features = self.dropout2(features)
        
        x = F.relu(self.fc1(torch.concat([x, features.squeeze(1)], axis=1)))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        return x
    


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = M5_fusion_all_transformer(n_input=1, n_output=2)

    dummy_input = torch.zeros((1, 1, 512)).to(device)
    dummy_input1 = torch.zeros((1, 1, 1)).to(device)
    dummy_input2 = torch.zeros((1, 1, 1)).to(device)
    dummy_input3 = torch.zeros((1, 1, 35)).to(device)
    model.eval()
    model.to(device)
    # print(model)
    # for name, p in model.named_parameters():
    #     if "fc1" in name:
    #         print(p)
    # n = count_parameters(model)
    # print("Number of parameters: %s" % n)

    reg = model(dummy_input, dummy_input1, dummy_input2, dummy_input3)
