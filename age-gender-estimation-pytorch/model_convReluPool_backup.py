import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
import torch.nn.functional as F
import torch.hub

# Only num_classes is used
def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = ageGenderNet(num_classes=num_classes)
    return model

# TODO: Add arguments (Net, pretrained)
class ageGenderNet(nn.Module):
    """
        Parameters: 
            num_classes: Age range
        return:
            age: 101 classes 
            gender: 2 classes
    """
    def __init__(self, num_classes=101):
        super(ageGenderNet, self).__init__()
        resnet = torch.hub.load(
            'moskomule/senet.pytorch',
            'se_resnet50',
            pretrained=True,)
    
        # self.model = nn.Sequential(nn.ModuleList(resnet.children())[:-1])    # Remove the last fc layer
        self.model = nn.Sequential(*list(resnet.children())[:-2])    # Remove the last fc layer

        self.conv_1 = nn.Conv2d(in_channels=resnet.fc.in_features, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), bias=False)        
        self.relu_1 = nn.ReLU() 
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.relu_2 = nn.ReLU() # activation
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        self.fc_1 = nn.Linear(512, 256)   
        self.age_cls_pred = nn.Linear(256, num_classes)    

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc_2 = nn.Linear(resnet.fc.in_features, 512)    # ([1000, 512])
        self.gen_cls_pred = nn.Linear(512, 2)    # ([512, 2])
    
    def get_age_gender(self, x):
        # x = self.resNet.avgpool(x)

        # x = x.view(x.size(0), -1)

        # age_pred = F.relu(self.conv_1(x))
        age_pred = self.conv_1(x)
        age_pred = self.relu_1(age_pred)
        age_pred = self.maxpool_1(age_pred) # [32, 1024, 3, 3]
        age_pred = self.conv_2(age_pred)
        age_pred = self.relu_2(age_pred)
        age_pred = self.maxpool_2(age_pred) # [32, 1024, 1, 1]
        age_pred = age_pred.view(age_pred.size(0), -1)
        age_pred = self.fc_1(age_pred)
        age_pred = self.age_cls_pred(age_pred)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        gen_pred = self.fc_2(x)
        gen_pred = self.gen_cls_pred(gen_pred)

        return age_pred, gen_pred

    def forward(self, x):
        # TODO: Error -> NotImplementedError
        x = self.model(x)
        age_pred, gen_pred = self.get_age_gender(x)
        return age_pred, gen_pred

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    model = get_model()
    print(model)
    print('count_parameters: ', count_parameters(model))


if __name__ == '__main__':
    main()
