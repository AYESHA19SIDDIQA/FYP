import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from IPython.display import clear_output
from torchmetrics import ROC, AUROC
import matplotlib.pyplot as plt


# adding gaze loss helper
def normalize_map(x, eps=1e-8):
    """Min-max normalize per sample to [0,1]. x shape: (B, C, T)"""
    x_min = x.amin(dim=(1,2), keepdim=True)
    x_max = x.amax(dim=(1,2), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)
def gaze_loss_fn(cam_pred, gaze_map, loss_type='mse'):
    """
    cam_pred: (B, channels, time) — model CAM for target class (unnormalized)
    gaze_map: (B, channels, time) — target human gaze heatmap (non-negative)
    loss_type: 'mse' or 'kl' (KL over flattened spatial-temporal distribution)
    Returns scalar loss (mean over batch)
    """
    # Normalize both to [0,1]
    cam_n = normalize_map(cam_pred)
    gaze_n = normalize_map(gaze_map)

    if loss_type == 'mse':
        return F.mse_loss(cam_n, gaze_n, reduction='mean')

    elif loss_type == 'kl':
        # Convert to distributions (softmax over channels*time)
        B = cam_n.shape[0]
        cam_flat = cam_n.view(B, -1)
        gaze_flat = gaze_n.view(B, -1)

        cam_p = F.log_softmax(cam_flat, dim=1)            # log P_pred
        gaze_q = F.softmax(gaze_flat, dim=1)              # Q_target

        # KL(Q || P) = sum Q * (log Q - log P)
        kl = torch.sum(gaze_q * (torch.log(gaze_q + 1e-12) - cam_p), dim=1)
        return kl.mean()

    else:
        raise ValueError("loss_type must be 'mse' or 'kl'")

#updating eval and train 

# -------------------------
# Updated evaluate: supports gaze logging (no grads)
# Expects val_loader batches: (data, target, gaze_map)
# -------------------------
def evaluate(model, val_loader, criterion, device, metrics, history, plot_roc=False, lambda_gaze=0.0, gaze_loss_type='mse'):
    model.to(device)
    val_loss = 0.0
    model.eval()
    metrics.reset()

    roc_curve = ROC(task='binary').to(device)
    auroc = AUROC(task='binary').to(device)

    actual = torch.tensor([], device=device, dtype=torch.long)
    pred   = torch.tensor([], device=device, dtype=torch.long)

    gaze_loss_total = 0.0
    gaze_batches = 0

    with torch.no_grad():
        for data, target, gaze_map in val_loader:
            data = data.to(device).float()
            target = target.to(device).float()
            gaze_map = gaze_map.to(device).float()   # shape (B, C, T)

            # model must return logits and cam_maps when asked
            outputs = model(data, return_cam=True)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                output, cam_maps = outputs
            elif isinstance(outputs, tuple) and len(outputs) == 3:
                # Some variants return (logits, cam, extra). handle safely:
                output, cam_maps, _ = outputs
            else:
                raise RuntimeError("Model must return (logits, cam_maps) when return_cam=True")

            # classification loss (criterion must accept logits and target)
            loss = criterion(output, target)
            val_loss += loss.item()

            # classification metrics
            _, predicted = output.max(1)
            label_check = target.argmax(1).long()
            metrics.update(label_check, predicted)
            actual = torch.cat([actual, label_check])
            pred   = torch.cat([pred, predicted])

            # ROC updates (use probability for positive class)
            probs = F.softmax(output, dim=1)
            roc_curve.update(probs[:, 1], label_check)
            auroc.update(probs[:, 1], label_check)

            # gaze loss logging (no grad)
            if lambda_gaze > 0.0:
                # cam_maps shape -> (B, num_classes, channels, time)
                # for binary task we take class index 1 (abnormal / positive)
                cam_pos = cam_maps[:, 1, :, :]   # (B, channels, time)
                g_loss = gaze_loss_fn(cam_pos, gaze_map, loss_type=gaze_loss_type)
                gaze_loss_total += g_loss.item()
                gaze_batches += 1

        val_loss /= len(val_loader)
        results = metrics.compute()
        results.update({"loss": val_loss})
        if gaze_batches > 0:
            results.update({"gaze_loss": gaze_loss_total / gaze_batches})
        history.update(results, 'val')

    history.update_cm(actual.tolist(), pred.tolist())

    if plot_roc:
        fpr, tpr, thresholds = roc_curve.compute()
        plt.figure()
        plt.plot(fpr.cpu(), tpr.cpu(), label=f'ROC (AUC = {auroc.compute():.2f})')
        plt.plot([0,1], [0,1], 'k--', label='Chance (AUC = 0.50)')
        plt.xlim(0,1); plt.ylim(0,1.05)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve'); plt.legend(loc='lower right'); plt.show()

    return val_loss

# -------------------------
# Updated train: expects train_loader batches: (data, target, gaze_map)
# Integrates gaze loss into total loss: total = cls_loss + lambda_gaze * gaze_loss
# -------------------------
def train(model, train_loader, val_loader, optimizer, criterion, epochs, history, metrics, device,
          save_path, earlystopping, accum_iter = 1, scheduler=None, save_best_acc=False,
          lambda_gaze=0.0, gaze_loss_type='mse'):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        metrics.reset()
        batch_idx = 0

        for data, target, gaze_map in tqdm(train_loader):
            data = data.to(device).float()
            target = target.to(device).float()
            gaze_map = gaze_map.to(device).float()

            # Forward with CAM output
            outputs = model(data, return_cam=True)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                output, cam_maps = outputs
            elif isinstance(outputs, tuple) and len(outputs) == 3:
                output, cam_maps, _ = outputs
            else:
                raise RuntimeError("Model must return (logits, cam_maps) when return_cam=True")

            # classification loss
            cls_loss = criterion(output, target)

            # gaze loss (take positive class = index 1 for abnormal)
            if lambda_gaze > 0.0:
                cam_pos = cam_maps[:, 1, :, :]   # (B, channels, time)
                g_loss = gaze_loss_fn(cam_pos, gaze_map, loss_type=gaze_loss_type)
            else:
                g_loss = torch.tensor(0.0, device=device)

            total_loss = cls_loss + lambda_gaze * g_loss

            # backprop (accumulation supported)
            total_loss.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            # metrics bookkeeping (use logits -> probs)
            with torch.no_grad():
                probs = F.softmax(output, dim=-1)
                _, predicted = torch.max(probs, 1)
                label_check = torch.argmax(target, 1)

            train_loss += total_loss.item()
            batch_idx += 1

            # free memory
            del data, target, output, cam_maps, cls_loss, total_loss
            if device == 'cuda':
                torch.cuda.empty_cache()

            metrics.update(label_check, predicted)

        # epoch finalize
        train_loss /= len(train_loader)
        results = metrics.compute()
        results.update({"loss": train_loss})
        history.update(results, 'train')

        # validation evaluation (pass lambda_gaze so validate can report gaze loss)
        val_loss = evaluate(model, val_loader, criterion, device, metrics, history,
                            plot_roc=False, lambda_gaze=lambda_gaze, gaze_loss_type=gaze_loss_type)

        clear_output(wait=True)

        # early stopping logic (same as your previous code)
        if save_best_acc:
            earlystopping(history.history["val"]["accuracy"][-1], model, save_best_acc=True)
        else:
            earlystopping(val_loss, model)

        if scheduler:
            scheduler.step(val_loss)

        if earlystopping.early_stop:
            print("Early stopping")
            break

        if device == 'cuda':
            torch.cuda.empty_cache()

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}', flush=True)
        print(f'Train Accuracy: {float(history.history["train"]["accuracy"][-1]):.4f} - Val Accuracy: {float(history.history["val"]["accuracy"][-1]):.4f}', flush=True)
        print(f'Train F1 Score: {float(history.history["train"]["f1score"][-1]):.4f} - Val F1 Score: {float(history.history["val"]["f1score"][-1]):.4f}', flush = True)
        history.print_best()
        history.plot()

    # load best model and final eval
    model.load_state_dict(torch.load(earlystopping.path))
    val_loss = evaluate(model, val_loader, criterion, device, metrics, history,
                        plot_roc=False, lambda_gaze=lambda_gaze, gaze_loss_type=gaze_loss_type)
    clear_output(wait=True)
    history.plot()
    history.print_best()
    history.display_cm()

    return model

# -------------------------
# Original evaluate and train functions (without gaze loss)
# ------------------------- 
# def evaluate(model, val_loader, criterion, device, metrics, history, plot_roc=False):
#     model.to(device)
#     val_loss = 0.0
#     model.eval()
#     metrics.reset()

#     # ROC metric for binary
#     roc_curve = ROC(task='binary').to(device)
#     auroc = AUROC(task='binary').to(device)

#     actual = torch.tensor([], device=device, dtype=torch.long)
#     pred   = torch.tensor([], device=device, dtype=torch.long)

#     with torch.no_grad():
#         for data, target in val_loader:
#             data, target = data.to(device).float(), target.to(device).float()
#             output = model(data).float()

#             # loss & metrics
#             loss = criterion(output, target)
#             val_loss += loss.item()

#             _, predicted = output.max(1)
#             label_check = target.argmax(1).long()
#             metrics.update(label_check, predicted)

#             actual = torch.cat([actual, label_check])
#             pred   = torch.cat([pred, predicted])

#             # ROC: use positive-class prob
#             auroc.update(predicted, label_check)
#             probs = F.softmax(output, dim=1)
#             roc_curve.update(probs[:, 1], label_check)

#         # finalize loss & metrics
#         val_loss /= len(val_loader)
#         results = metrics.compute()
#         results.update({"loss": val_loss})
#         history.update(results, 'val')

#     history.update_cm(actual.tolist(), pred.tolist())

#     if plot_roc:
#         # compute ROC curve data
#         fpr, tpr, thresholds = roc_curve.compute()
#         # print(f"fpr: {fpr}")
#         # print(f"tpr: {tpr}")
#         # print(f"thresholds: {thresholds}")

#         # plot
#         plt.figure()
#         plt.plot(fpr.cpu(), tpr.cpu(), label=f'ROC (AUC = {auroc.compute():.2f})')
#         plt.plot([0,1], [0,1], 'k--', label='Chance (AUC = 0.50)')
#         plt.xlim(0,1); plt.ylim(0,1.05)
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('ROC Curve')
#         plt.legend(loc='lower right')
#         plt.show()

#     return val_loss

# def train(model, train_loader, val_loader, optimizer, criterion, epochs, history, metrics, device, save_path, earlystopping, accum_iter = 1, scheduler=None, save_best_acc=False):
#     model = model.to(device)
#     model.train()
#     for epoch in range(epochs):
#         train_loss = 0
#         metrics.reset()
#         batch_idx = 0
#         for data, target in tqdm(train_loader):
#             data, target = data.to(device), target.to(device)
#             data = data.float()
#             target = target.float()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
#                 optimizer.step()
#                 optimizer.zero_grad()
#             output = F.softmax(output, dim = -1)
#             _, predicted = torch.max(output, 1)
#             label_check = torch.argmax(target, 1)
#             train_loss += loss.item()
#             batch_idx += 1
#             # clearing data for space
#             del data, target, output, loss
#             if device == 'cuda':
#                 torch.cuda.empty_cache()
#             metrics.update(label_check, predicted)
#         train_loss /= len(train_loader)
#         results = metrics.compute()
#         results.update({"loss": train_loss})
#         history.update(results, 'train')
#         val_loss = evaluate(model, val_loader, criterion, device, metrics, history)
#         model.train()
#         clear_output(wait=True)
#         if save_best_acc:
#             earlystopping(history.history["val"]["accuracy"][-1], model, save_best_acc=True)
#             if scheduler:
#                 scheduler.step(val_loss)
#             if earlystopping.early_stop:
#                 print("Early stopping")
#                 break 
#         else:
#             earlystopping(val_loss, model)
#             if scheduler:
#                 scheduler.step(val_loss)
#             if earlystopping.early_stop:
#                 print("Early stopping")
#                 break
#         if device == 'cuda':
#             torch.cuda.empty_cache()
#         print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}', flush=True)
#         print(f'Train Accuracy: {float(history.history["train"]["accuracy"][-1]):.4f} - Val Accuracy: {float(history.history["val"]["accuracy"][-1]):.4f}', flush=True)
#         print(f'Train F1 Score: {float(history.history["train"]["f1score"][-1]):.4f} - Val F1 Score: {float(history.history["val"]["f1score"][-1]):.4f}', flush = True)
#         history.print_best()
#         history.plot()
#     model.load_state_dict(torch.load(earlystopping.path))
#     val_loss = evaluate(model, val_loader, criterion, device, metrics, history)
#     clear_output(wait=True)
#     history.plot()
#     history.print_best()
#     history.display_cm()
#     return model
