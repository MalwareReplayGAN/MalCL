import torch
import torch.nn as nn
import torch.nn.functional as F
from data_ import get_ember_test_data

use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()

data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
X_test, Y_test, Y_test_onehot = get_ember_test_data(data_dir)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        self.input_channel = 1
        self.channel_b = 128
        self.channel_c = 256
        self.channel_d = 512
        self.output_dim = 100
        self.drop_prob = 0.3

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_channel, self.channel_d, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.channel_d),
            nn.Dropout(self.drop_prob),
            nn.Conv1d(self.channel_d, self.channel_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.channel_c),
            nn.Dropout(self.drop_prob),
            nn.Conv1d(self.channel_c, self.channel_b, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.channel_b),
            nn.Dropout(self.drop_prob),
            nn.Flatten())

        self.fc = nn.Linear(self.channel_b * self.input_features, self.output_dim)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, self.input_channel, self.input_features)
        x = self.conv(x)
        x = x.view(-1, self.channel_b * self.input_features)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def predict(self, x_data):
        x_data = self.forward(x_data)
        result = self.softmax(x_data)
        return result

PATH = "/home/02mjpark/ConvGAN/SAVE/bsmdl.pt"
# model = torch.load(PATH)
model = Classifier()
model.load_state_dict(torch.load(PATH))

def test(model, x_test, y_test, Y_test_onehot):

    x_test = torch.FloatTensor(x_test)
    y_test = torch.Tensor(y_test)
    y_test = y_test.float()

    Y_test_onehot = torch.Tensor(Y_test_onehot)
    Y_test_onehot = Y_test_onehot.float()

    # print(x_test.shape)
    # print(y_test.shape)
    # use_cuda = False
    # if use_cuda:
    #     x_test = x_test.cuda(0)
    #     # y_test = y_test.cuda(0)
    #     model = model.cuda(0)

    model.eval()
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, Y_test_onehot)

    print('Accuracy: {}% Cost: {:.6f}'.format(
        correct_count / len(y_test) * 100, cost.item()
    ))    

with torch.no_grad():
    test(model, X_test, Y_test, Y_test_onehot)
