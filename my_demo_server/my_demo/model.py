import torch.nn as nn
import torch.nn.functional as F
import torch.hub

# Only num_classes is used
def get_model(model_name="efficientnet_lite0", num_classes=101, pretrained=True):
    model = ageGenderNet(num_classes=num_classes)
    return model

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
        efficient_net = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_lite0', pretrained=True)
        # self.model = nn.Sequential(*list(efficient_net.children())[:-1])
        self.model = nn.Sequential(*list(efficient_net.children())[:-2])    # Remove the last fc layer
        self.avgpool_age = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool_gen = nn.AdaptiveAvgPool2d(output_size=1)

        self.age_conv = nn.Sequential(
            nn.Conv2d(in_channels=efficient_net.classifier.in_features, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),

            # nn.Sigmoid()
        )
        self.age_fc = nn.Sequential(
            nn.Linear(512, num_classes)            
        )

        self.gen_model = nn.Sequential(
            nn.Linear(efficient_net.classifier.in_features, 128),
            nn.ReLU(), 
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    # x: torch.Size([32, 1280, 7, 7])
    def get_age_gender(self, x):
        # flatten
        # x = x.view(x.size(0), -1)
        
        # age
        age = self.age_conv(x) # age: torch.Size([32, 512, 3, 3])
        age = self.avgpool_age(age) # age: torch.Size([32, 1280, 1, 1])
        age = age.view(age.size(0), -1)
        age = self.age_fc(age)
        
        # gender
        x = self.avgpool_gen(x) # x: torch.Size([32, 1280, 1, 1])
        gen = x.view(x.size(0), -1) # gen: torch.Size([32, 1280,])
        gen = self.gen_model(gen)

        return age, gen

    def forward(self, x):
        x = self.model(x)
        age_pred, gen_pred = self.get_age_gender(x)
        return age_pred, gen_pred

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = get_model()
    print(model)
    print('count_parameters: ', count_parameters(model))