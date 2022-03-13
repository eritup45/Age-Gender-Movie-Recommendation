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
        self.model = nn.Sequential(*list(efficient_net.children())[:-1])

        self.fc_age_1 = nn.Linear(efficient_net.classifier.in_features, 1024)
        self.relu_age_1 = nn.ReLU()
        self.fc_age_2 = nn.Linear(1024, 512)
        self.relu_age_2 = nn.ReLU()
        self.fc_age_3 = nn.Linear(512,num_classes)

        # self.fc_gen_1 = nn.Linear(efficient_net.classifier.in_features, 1024)
        # self.relu_gen_1 = nn.ReLU()
        # self.fc_gen_2 = nn.Linear(1024, 2)

        self.age_model = nn.Sequential(
            nn.Linear(efficient_net.classifier.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        self.gen_model = nn.Sequential(
            nn.Linear(efficient_net.classifier.in_features, 128),
            nn.ReLU(), 
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    def get_age_gender(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        
        # age
        age = self.age_model(x)
        # age = self.fc_age_1(x)
        # age = self.relu_age_1(age)
        # age = self.fc_age_2(age)
        # age = self.relu_age_2(age)
        # age = self.fc_age_3(age)
        
        # gender
        gen = self.gen_model(x)
        # gen = self.fc_gen_1(x)
        # gen = self.relu_gen_1(gen)
        # gen = self.fc_gen_2(gen)

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