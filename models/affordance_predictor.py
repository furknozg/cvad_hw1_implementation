import torch.nn as nn


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
	resnet = models.resnet18(pretrained=True, requires_grad=True)
	self.feature_ex = nn.Sequential(resnet)
	
	self.lanedist_fc = nn.Sequential(
                nn.Linear(1,128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.Dropout(0.5),
                nn.ReLU(),

        )
	self.routeangle_fc = nn.Sequential(
                nn.Linear(1,128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.Dropout(0.5),
                nn.ReLU(),

	)

	self.TLdist_fc = nn.Sequential(
                nn.Linear(1,128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.Dropout(0.5),
                nn.ReLU(),
	
	
	)

	self.TLstate_fc = nn.Sequential(
		nn.Linear(512, 256),
		nn.Dropout(0.5),
		nn.Linear(256,128),
		nn.Dropout(0.5),
		nn.ReLU(),
		nn.Linear(128, 3) # For 3 states that are possible for TL
		nn.Softmax(dim = 3)
	)
	

    def forward(self, img):
        img = self.feature_ex(img)

	lanedist = self.lanedist_fc(img)
	angle = self.routeangle_fc(img)
	TLdist = self.TLdist_fc(img)
	TLstate = torch.argmax(self.TLstate_fc(img))
	return lanedist, angle, TLdist, TLstate
