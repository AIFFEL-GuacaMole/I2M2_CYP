# train.py

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, average_precision_score
from utils.loss_fn import CustomLoss
from utils.earlystop import EarlyStopping
from training_structures.unimodal import get_unimodal_logits
from training_structures.intra_modality import get_intra_logits
from training_structures.inter_modality import get_inter_logits
from training_structures.inter_and_intra_modality import get_inter_intra_logits

def train_model(model_or_tuple, train_loader, val_loader, mode, args, device="cuda"):
    wandb.init(project="I2M2", config=vars(args))

    criterion = CustomLoss(args.task_type)
    stopper = EarlyStopping(patience=args.patience, verbose=True)
    best_val_loss = float('inf')
    best_state = None

    if mode=="unimodal":
        model = model_or_tuple.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    elif mode=="intra":
        m1, m2, m3 = model_or_tuple
        m1.to(device); m2.to(device); m3.to(device)
        opt1 = optim.Adam(m1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt2 = optim.Adam(m2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt3 = optim.Adam(m3.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched1 = ReduceLROnPlateau(opt1, mode="min", factor=0.5, patience=3, verbose=True)
        sched2 = ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=3, verbose=True)
        sched3 = ReduceLROnPlateau(opt3, mode="min", factor=0.5, patience=3, verbose=True)

    else: 
        model = model_or_tuple.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    for epoch in range(args.epochs):
        if mode=="unimodal":
            model.train()
        elif mode=="intra":
            (m1,m2,m3) = model_or_tuple
            m1.train(); m2.train(); m3.train()
        else: 
            model_or_tuple.train()

        running_loss=0.0
        running_count=0
        train_labels = []
        train_probs  = []
        train_targets= []

        for batch in tqdm(train_loader, desc=f"[{mode} Train] E{epoch+1}"):
            if mode=="unimodal":
                optimizer.zero_grad()
                logits = get_unimodal_logits(model, batch, device, args.model_type)
                y = batch["y"].to(device)
                if args.task_type=="classification":
                    y = y.long()
                else:
                    y = y.float()
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            elif mode=="intra":
                opt1.zero_grad(); opt2.zero_grad(); opt3.zero_grad()
                (m1,m2,m3) = model_or_tuple
                logits = get_intra_logits(m1,m2,m3,batch,device)
                y = batch["y"].to(device)
                if args.task_type=="classification":
                    y = y.long()
                else:
                    y = y.float()
                loss = criterion(logits, y)
                loss.backward()
                opt1.step(); opt2.step(); opt3.step()

            elif mode=="inter":
                optimizer.zero_grad()
                logits = get_inter_logits(model_or_tuple, batch, device)
                y = batch["y"].to(device)
                if args.task_type=="classification":
                    y = y.long()
                else:
                    y = y.float()
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            else:
                optimizer.zero_grad()
                logits = get_inter_intra_logits(model_or_tuple,batch,device)
                y = batch["y"].to(device)
                if args.task_type=="classification":
                    y = y.long()
                else:
                    y = y.float()
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            running_count+= bs

            if args.task_type=="classification":
                import torch.nn.functional as F
                _, pred_label = torch.max(logits, dim=1)
                train_labels.extend(pred_label.cpu().numpy())
                soft = F.softmax(logits, dim=1)
                pos_prob = soft[:,1].detach().cpu().numpy()
                train_probs.extend(pos_prob)
                train_targets.extend(y.cpu().numpy())
            else:
                train_labels.extend(logits.detach().cpu().numpy())
                train_targets.extend(y.cpu().numpy())

        train_loss = running_loss / running_count

        if args.task_type=="classification":
            train_acc = accuracy_score(train_targets, train_labels)
        else:
            train_mse = mean_squared_error(train_targets, train_labels)

        val_loss, val_m1, val_m2 = evaluate_model(model_or_tuple, val_loader, mode, args, device)

        if mode=="unimodal":
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']

        elif mode=="intra":
            sched1.step(val_loss)
            sched2.step(val_loss)
            sched3.step(val_loss)

            lr1 = opt1.param_groups[0]['lr']
            lr2 = opt2.param_groups[0]['lr']
            lr3 = opt3.param_groups[0]['lr']
            avg_lr = (lr1 + lr2 + lr3) / 3.0

        else: 
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

        stopper(val_loss, None)
        if stopper.early_stop:
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if mode=="unimodal":
                best_state = model.state_dict()
            elif mode=="intra":
                (m1,m2,m3) = model_or_tuple
                best_state = (m1.state_dict(), m2.state_dict(), m3.state_dict())
            else:
                best_state = model_or_tuple.state_dict()

        if args.task_type=="classification":
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
                  f"ValLoss={val_loss:.4f}, ValAcc={val_m1:.4f}, ValAUPRC={val_m2:.4f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"TrainLoss={train_loss:.4f}, TrainMSE={train_mse:.4f} | "
                  f"ValLoss={val_loss:.4f}, ValMSE={val_m1:.4f}, ValMAE={val_m2:.4f}")
        
        if mode=="intra" :
            if args.task_type=="classification":
                wandb.log({"epoch": epoch+1, "learning_rate_avg": avg_lr, "train_loss": train_loss, 
                       "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_m1, "val_auprc": val_m2})
            elif args.task_type=="regression" :
                wandb.log({"epoch": epoch+1, "learning_rate_avg": avg_lr, "train_loss": train_loss, 
                        "train_mse": train_mse, "val_loss": val_loss, "val_mse": val_m1, "val_mae": val_m2})
        else:
            if args.task_type=="classification":
                wandb.log({"epoch": epoch+1, "learning_rate": current_lr, "train_loss": train_loss, 
                       "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_m1, "val_auprc": val_m2})
            elif args.task_type=="regression":
                wandb.log({"epoch": epoch+1, "learning_rate": current_lr, "train_loss": train_loss, 
                        "train_mse": train_mse, "val_loss": val_loss, "val_mse": val_m1, "val_mae": val_m2})
 
    if best_state is not None:
        if mode=="unimodal":
            model_or_tuple.load_state_dict(best_state)
            ckpt_path=f"./ckpt/best_unimodal_{args.model_type}_{args.task_type}.pth"
            torch.save(model_or_tuple.state_dict(), ckpt_path)
        elif mode=="intra":
            (m1,m2,m3) = model_or_tuple
            m1.load_state_dict(best_state[0])
            m2.load_state_dict(best_state[1])
            m3.load_state_dict(best_state[2])
            torch.save(m1.state_dict(), f"./ckpt/best_intra_1d_{args.task_type}.pth")
            torch.save(m2.state_dict(), f"./ckpt/best_intra_2d_{args.task_type}.pth")
            torch.save(m3.state_dict(), f"./ckpt/best_intra_3d_{args.task_type}.pth")
        elif mode=="inter":
            model_or_tuple.load_state_dict(best_state)
            ckpt_path=f"./ckpt/best_inter_{args.task_type}.pth"
            torch.save(model_or_tuple.state_dict(), ckpt_path)
        else:
            model_or_tuple.load_state_dict(best_state)
            ckpt_path=f"./ckpt/best_inter_intra_{args.task_type}.pth"
            torch.save(model_or_tuple.state_dict(), ckpt_path)
    wandb.finish()

def evaluate_model(model_or_tuple, val_loader, mode, args, device="cuda"):
    import torch.nn.functional as F
    model_or_tuple.eval() if mode!="intra" else None
    if mode=="intra":
        (m1,m2,m3)=model_or_tuple
        m1.eval(); m2.eval(); m3.eval()
    crit = CustomLoss(args.task_type)
    total_loss=0.0
    total_count=0
    all_labels=[]
    all_probs=[]
    all_targs=[]

    with torch.no_grad():
        for batch in val_loader:
            if mode=="unimodal":
                logits = get_unimodal_logits(model_or_tuple,batch,device,args.model_type)
            elif mode=="intra":
                (m1,m2,m3)=model_or_tuple
                logits = get_intra_logits(m1,m2,m3,batch,device)
            elif mode=="inter":
                logits = get_inter_logits(model_or_tuple,batch,device)
            else:
                logits = get_inter_intra_logits(model_or_tuple,batch,device)
            y = batch["y"].to(device)
            if args.task_type=="classification":
                y=y.long()
            else:
                y=y.float()
            loss=crit(logits,y)
            bs=y.size(0)
            total_loss+=loss.item()*bs
            total_count+=bs
            if args.task_type=="classification":
                _, pred_label = torch.max(logits,dim=1)
                all_labels.extend(pred_label.cpu().numpy())
                soft = F.softmax(logits, dim=1)
                pos_prob = soft[:,1].detach().cpu().numpy()
                all_probs.extend(pos_prob)
                all_targs.extend(y.cpu().numpy())
            else:
                all_labels.extend(logits.detach().cpu().numpy())
                all_targs.extend(y.cpu().numpy())

    val_loss= total_loss/total_count
    if args.task_type=="classification":
        val_acc = accuracy_score(all_targs, all_labels)
        val_auprc = average_precision_score(all_targs, all_probs)
        return val_loss, val_acc, val_auprc
    else:
        val_mse= mean_squared_error(all_targs, all_labels)
        val_mae= mean_absolute_error(all_targs, all_labels)
        return val_loss, val_mse, val_mae