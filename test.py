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

        self.fc1 = nn.Linear(self.input_features, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc1_drop = nn.Dropout(self.drop_prob)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc2_drop = nn.Dropout(self.drop_prob)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc3_drop = nn.Dropout(self.drop_prob)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(256, self.output_dim)
        self.fc4_bn = nn.BatchNorm1d(self.output_dim)
        self.fc4_drop = nn.Dropout(self.drop_prob)
        self.act4 = nn.ReLU()

        # self.fc4 = nn.Linear(256, 128)
        # self.fc4_bn = nn.BatchNorm1d(128)
        # self.fc4_drop = nn.Dropout(0.5)
        # self.act4 = nn.ReLU()


        # self.fc5 = nn.Linear(128, self.output_dim)
        # self.fc5_bn = nn.BatchNorm1d(self.output_dim)
        # self.fc5_drop = nn.Dropout(self.drop_prob)
        # self.act5 = nn.ReLU()

        # self.fc2 = nn.Linear(self.input_features, self.output_dim)

        self.softmax = nn.Softmax()

    def forward(self, x):

        x = x.view(-1, self.input_features)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_drop(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.fc2_drop(x)
        x = self.act2(x)

        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = self.fc3_drop(x)
        x = self.act3(x)

        x = self.fc4(x)
        x = self.fc4_bn(x)
        x = self.fc4_drop(x)
        x = self.act4(x)

        # x = self.fc5(x)
        # x = self.fc5_bn(x)
        # x = self.fc5_drop(x)
        # x = self.act5(x)

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
