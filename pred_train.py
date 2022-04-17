import torch
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor


ptimizer = torch.optim.SGD(AffordancePredictor().parameters(), lr=0.001, momentum=0.9)

loss_fn = torch.nn.CrossEntropyLoss()

def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    # Your code here

        valid_loss = 0.0
        model.eval()
        for data, labels in validloader:
                if torch.cuda.is_available():
                         data, labels = data.cuda(), labels.cuda()
                         target = model(data)
                         loss = criterion(target,labels)
                         valid_loss += loss.item()
        return valid_loss



def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
	global optimizer
    	last_loss = 0
	for i, data in enumerate(dataloader)
		imgs = data[:,0]
                obj_not = data[:,1]
                lane_dist = obj_not[:,7]
                rout_angl = obj_not[:,6]
		TL_dist = obj_not[:,12]
		TL_state = obj_not[:,11]
                outputs = model(imgs)

                optimizer.zero_grad()
                loss = loss_fn([outputs[0], outputs[1], outputs[2], outputs[3]], [lane_dist, rout_angl, TL_dist, TL_state])
                loss.backward()

                optimizer.step()
                last_loss += loss
        return last_loss


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
	
        num_epoch_arr = np.array(range(0, len(train_loss)))

        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.plot(train_loss, num_epoch_arr, color = "red", marker = "o", label = "train")
        plt.plot(val_loss, num_epoch_arr, color = "green", marker = "o", label = "validation")
        plt.legend()
        plt.show()




def main():

    # Change these paths to the correct paths in your downloaded expert dataset
    
    train_root = "/userfiles/eozsuer16/expert_data/train"
    val_root = "/userfiles/eozsuer16/expert_data/val"

    model = AffordancePredictor()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
