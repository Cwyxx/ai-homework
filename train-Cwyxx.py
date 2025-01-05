import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import wandb

# construct dataset.
class Cwyxx_Dataset(torch.utils.data.Dataset):
    def __init__(self, tensor_1_path_list, tensor_2_path_list, label_list):
        super(Cwyxx_Dataset, self).__init__()
        self.tensor_tuple_list = []
        # print(f"tensor_1_path_list:{len(tensor_1_path_list)}")
        sum_label = 0
        for tensor_1_path, tensor_2_path, label in zip(tensor_1_path_list, tensor_2_path_list, label_list):
            tensor_1_basename = os.path.basename(tensor_1_path).replace("_semantic_content.pt", "")
            tensor_2_basename = os.path.basename(tensor_2_path).replace("_coherence.pt", "")
            if tensor_1_basename == tensor_2_basename:
                self.tensor_tuple_list.append((tensor_1_path, tensor_2_path, label))
                sum_label += label
            else:
                print("WDF-Dataset")
                exit(0)
        
        random.shuffle(self.tensor_tuple_list)
        print(f"Construct Dataset: {len(self.tensor_tuple_list)}, Fake Image Num = {sum_label}")
        
    def __len__(self):
        return len(self.tensor_tuple_list)

    def __getitem__(self, idx):
        tensor_1_path, tensor_2_path, label = self.tensor_tuple_list[idx]
        tensor_1 = torch.load(tensor_1_path).to(torch.float)
        tensor_2 = torch.load(tensor_2_path).to(torch.float)
        return tensor_1, tensor_2, label
    
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
class Cwyxx_Net(nn.Module):
    def __init__(self):
        super(Cwyxx_Net, self).__init__()
        self.drop = 0.1
        
        self.tensor_1_process = nn.Sequential(
            nn.Linear(4096, 784),
            nn.ReLU(),
            nn.Dropout(self.drop)
        )
        
        self.tensor_2_process = nn.Sequential(
            nn.Linear(4096, 784),
            nn.ReLU(),
            nn.Dropout(self.drop)
        )
        
        self.total_tensor_process = nn.Sequential(
            nn.Linear(1568, 256),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(256, 1)
        )
        
    def forward(self, tensor_1, tensor_2):
        # print(type(tensor_1))
        # print(f"Before processing: {tensor_1.shape}, {tensor_2.shape}")
        tensor_1 = self.tensor_1_process(tensor_1) #[batch_size, feature]
        # print(f"tensor_1.shape:{tensor_1.shape}")
        tensor_2 = self.tensor_2_process(tensor_2) #[batch_size, feature]
        # print(f"tensor_2.shape:{tensor_2.shape}")
        total_tensor = torch.cat([tensor_1, tensor_2], dim=1) # [bacth_size, feature + feature]
        # print(f"total_tensor.shape:{total_tensor.shape}")
        score = self.total_tensor_process(total_tensor)
        return score
        

if __name__ == '__main__':
    seed = 1    
    lr = 0.0001
    batch_size = 32
    train_epoch_num = 15
    model_save_path = "/data3/chenweiyan/aigc/detection-method/ckpt/ckpt_homework/model_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(seed)
    
    wandb.init(
            # set the wandb project where this run will be logged
            project="Homework",

            # track hyperparameters and run metadata
            config={
            "seed" : seed,
            "learning_rate": lr,
            "architecture": "MLP",
            "dataset": "genimage",
            "batch_size" : batch_size,
            "epochs": train_epoch_num,
            }
        )
    
    def construct_dataset(base_tensor_root):
        image_type_list = ["0_real", "1_fake"]
        image_type_dict = {"0_real" : 0, "1_fake" : 1}
        tensor_1_path_list = []
        tensor_2_path_list = []
        label_list = []
        
        for image_type in image_type_list:
            tensor_root = os.path.join(base_tensor_root, image_type)
            tmp_tensor_1_path_list = []
            tmp_tensor_2_path_list = []
            tmp_label_list = []
            for tensor_name in os.listdir(tensor_root):
                if tensor_name.endswith("_semantic_content.pt"):
                    tmp_tensor_1_path_list.append(os.path.join(tensor_root, tensor_name))
                    tmp_label_list.append(image_type_dict[image_type])
                elif tensor_name.endswith("_coherence.pt"):
                    tmp_tensor_2_path_list.append(os.path.join(tensor_root, tensor_name))
                else:
                    print("WDF!")
                    exit(0)
            
            print(f"Loading tensor from {tensor_root} : {len(tmp_tensor_1_path_list)}, label: {image_type_dict[image_type]}")
            tmp_tensor_1_path_list = sorted(tmp_tensor_1_path_list)
            tmp_tensor_2_path_list = sorted(tmp_tensor_2_path_list)
            tensor_1_path_list.extend(tmp_tensor_1_path_list)
            tensor_2_path_list.extend(tmp_tensor_2_path_list)
            label_list.extend(tmp_label_list)

        assert len(tensor_1_path_list) == len(tensor_2_path_list) == len(label_list)
        # print(f"tensor_1_path_list:{len(tensor_1_path_list)}")
        dataset = Cwyxx_Dataset(tensor_1_path_list, tensor_2_path_list, label_list)
        return dataset
    
    # train-tensor
    print("Construct train dataset......")
    base_train_tensor_root = "/data3/chenweiyan/dataset/aigc-detection-dataset/genimage/sd_1.4/tensor-root/train"
    train_dataset = construct_dataset(base_train_tensor_root)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    
    # test-tensor
    print("Construct val dataset......")
    base_val_tensor_root = "/data3/chenweiyan/dataset/aigc-detection-dataset/genimage/sd_1.4/tensor-root/val"
    val_dataset = construct_dataset(base_val_tensor_root)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    
    # define model
    model = Cwyxx_Net()
    model.to(device)
    
    #==============={optimizer and loss function}=================
    bce_loss_function = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    #==============={accuracy-record}=============================
    record_accuracy = 0
    
    for epoch_index in range(train_epoch_num):
        epoch_loss_list = []
        model.train()
        for batch_tensor_1, batch_tensor_2, batch_label in tqdm(train_dataloader, desc=f"Epoch / train / {epoch_index}"):
            batch_label = batch_label.unsqueeze(1).to(torch.float).to(device)
            batch_tensor_1 = batch_tensor_1.squeeze(1).to(device) # [batch_size, feature_dim]
            batch_tensor_2 = batch_tensor_2.squeeze(1).to(device) # [batch_size, feature_dim]
            output_scores = model(batch_tensor_1, batch_tensor_2)
            optimizer.zero_grad()
            loss_ = bce_loss_function(output_scores, batch_label)
            loss_.backward()
            optimizer.step()
            epoch_loss_list.append(loss_.item())

        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for batch_tensor_1, batch_tensor_2, batch_label in tqdm(val_dataloader, desc=f"Epoch / val / {epoch_index}"):
                y_true.extend(batch_label.flatten().tolist())
                
                batch_label = batch_label.unsqueeze(1).to(torch.float).to(device)
                batch_tensor_1 = batch_tensor_1.squeeze(1).to(device) # [batch_size, feature_dim]
                batch_tensor_2 = batch_tensor_2.squeeze(1).to(device) # [batch_size, feature_dim]
                output_scores = model(batch_tensor_1, batch_tensor_2)
                
                y_pred.extend(output_scores.sigmoid().flatten().tolist())
            
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            real_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
            fake_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
            total_acc = accuracy_score(y_true, y_pred > 0.5)
            
            print(f"Epoch {epoch_index:01d} validation: Real Image Accuracy : {real_acc*100:.2f}, Fake Image Accuracy : {fake_acc*100:.2f}, Total Image Accuracy : {total_acc*100:.2f}")
            # print(f"Real Image Num: {len(y_true[y_true == 0])}, Fake Image Num: {len(y_true[y_true == 1])}")
            if fake_acc > record_accuracy:
                fake_acc = record_accuracy
                torch.save(model.state_dict(), model_save_path)
        
        wandb.log({"train_loss" : sum(epoch_loss_list)/len(epoch_loss_list), "val_real_acc" : real_acc, "val_fake_acc" : {fake_acc}, "total_acc" : total_acc})
    
    wandb.finish()
