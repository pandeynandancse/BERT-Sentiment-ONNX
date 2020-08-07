import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda import amp  #this is required if using autoatic mixed precision

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler, scaler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

	#move everything to appropriate device
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()

	with amp.autocast():   #from torch.cuda import amp   #this is required if using autoatic mixed precision
	
	#pass through model
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)	
            loss = loss_fn(outputs, targets)
	
        loss.backward() # replace this line with scaler.scale(loss).backward() ==>> #this is required if using autoatic mixed precision
        optimizer.step()  # replace this line with scaler.step(optimizer)  ==> #this is required if using autoatic mixed precision   
	scaler.update()   #this is required if using autoatic mixed precision else not required
        scheduler.step()    


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            
	    #torch.sigoid for outputs because we have linear layer
 	    fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
