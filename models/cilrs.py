import torch.nn as nn
import torchvision.models as models

class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CILRS,self).__init__()
	# Dimensions of the input network may be wrong, work on it later.
	resnet = models.resnet18(pretrained=True, requires_grad=False)
	
	self.feature_ex = nn.Sequential(resnet) 
	
	self.speed_fc = nn.Sequential(
		nn.Linear(1,128),
		nn.Dropout(0.5),
		nn.ReLU(),
		nn.Linear(128,128),
		nn.Dropout(0.5),
		nn.ReLU(),
	)

	self.embed_fc = nn.Sequential(
		nn.Linear(512 + 128, 512),
		nn.Dropout(0.5),
		nn.ReLU(),
	)
	
	self.cond_branch = nn.ModuleList([
		nn.Sequential(
			nn.Linear(512,256),
			nn.Dropout(0.5),
			nn.ReLU(),
			nn.Linear(256,256),
			nn.ReLU(),
			nn.Linear(256,3),
		) for i in range(4)
		#For the 4 commands
	])
	
	self.speed_branch = nn.Sequential(
		nn.Linear(512, 256),
		nn.Dropout(0.5),
		nn.ReLU(),
		nn.Linear(256,256),
		nn.ReLU(),
		nn.Linear(256,1),
	
	)

	self.com_fc = nn.Sequential()
    def forward(self, img, command):
	img = self.feature_ex(img)
	pred_speed = self.speed_branch(img)
	
	embedded = torch.cat([img,speed],dim = 1)
	embedded = self.embed_fc(embedded)
	
	control = torch.cat(
		[out(embedded) for out in self.cond_branch],dim=1)
	return control, pred_speed, img, embedded
	
