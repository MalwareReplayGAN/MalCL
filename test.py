import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from data_ import get_ember_test_data


use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()

data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
X_test, Y_test, Y_test_onehot = get_ember_test_data(data_dir)
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        # self.input_channel = 1
        self.output_dim = 100
        self.drop_prob = 0.5

        self.block1 = nn.Sequential(
            nn.Conv1d(self.input_features, 1024, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 512, 3, 3, 1),
            nn.BatchNorm1d(512),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
            nn.MaxPool1d(3, 3, 1)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
            nn.MaxPool1d(3, 3, 1)
        )
        
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.Dropout(self.drop_prob),
            nn.ReLU()
        )

        self.softmax = nn.Softmax()

    def forward(self, x):
        
        x = x.view(-1, self.input_features, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc1(x)

        x = self.softmax(x)
        return x

    def predict(self, x_data):
        x_data = self.forward(x_data)
        result = self.softmax(x_data)
        return result 


PATH = "/home/02mjpark/ConvGAN/SAVE/bsmdl.pt"
# model = torch.load(PATH)
model = Classifier()
saved_checkpoint = torch.load(PATH)
model.load_state_dict(saved_checkpoint, strict=False)

scaler = joblib.load('/home/02mjpark/ConvGAN/SAVE/scaler1.pkl')

def test(model, scaler, x_test, y_test, Y_test_onehot):

    x_test = scaler.transform(x_test)
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
    print(prediction.shape, Y_test_onehot.shape)

    # print(Y_test[1])
    # print(Y_test_onehot[1])
    # print(predicted_classes[1])
    print('Accuracy: {}% Cost: {:.6f}'.format(
        correct_count / len(y_test) * 100, cost.item()
    ))    

with torch.no_grad():
    test(model, scaler, X_test, Y_test, Y_test_onehot)
