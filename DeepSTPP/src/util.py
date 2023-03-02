import torch
import numpy as np
import copy

from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate
from comet_ml import Experiment

# # Create an experiment with your api key
experiment = Experiment(
    api_key="jap0PWqwWCbPCum539y0HzWFO",
    project_name="deepstpp",
    workspace="arnau01",
)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_diff(outputs, targets, portion=1):
    outputs = outputs[:int(len(outputs) * portion)]
    targets = targets[:int(len(targets) * portion)]

    plt.figure(figsize=(14, 10), dpi=180)
    plt.subplot(2, 2, 1)

    n = outputs.shape[0]
    lookahead = outputs.shape[1]

    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 0], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 0], "-o", color="b", label="Actual")
    plt.ylabel('Latitude')
    plt.legend()

    plt.subplot(2, 2, 2)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 1], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 1], "-o", color="b", label="Actual")
    plt.ylabel('Longitude')
    plt.legend()

    plt.subplot(2, 2, 3)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 2], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 2], "-o", color="b", label="Actual")
    plt.ylabel('delta_t (hours)')
    plt.legend()
    plt.savefig('result.png')


def eval_loss(model, test_loader, device):
    model.eval()
    loss_total = 0
    sll_meter = AverageMeter()
    tll_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    for index, data in enumerate(test_loader):
        st_x, st_y, _, _, _ = data
        #st_x, st_y, _, _ = data
        
        loss, sll, tll = model.loss(st_x, st_y)
        
        loss_meter.update(loss.item())
        sll_meter.update(sll.mean())
        tll_meter.update(tll.mean())
        
    return loss_meter.avg, sll_meter.avg, tll_meter.avg


def eval_loss_rmtpp(model, test_loader, device):
    model.eval()
    loss_total = 0
    loss_meter = AverageMeter()
    
    for index, data in enumerate(test_loader):
        st_x, st_y, _, _, _ = data
        loss = model.loss(st_x, st_y)
        
        loss_meter.update(loss.item())
        
    return loss_meter.avg


def train_rmtpp(model, train_loader, val_loader, config, logger, device):
    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=50, gamma=0.2)
    best_eval = np.infty
    loss_meter = AverageMeter()
    
    for epoch in trange(config.epochs):
        loss_total = 0
        model.train()
        for index, data in tenumerate(train_loader):
            st_x, st_y, _, _, _ = data

            model.optimizer.zero_grad()
            loss = model.loss(st_x, st_y)
            
            if torch.isnan(loss):
                print("Numerical error, quiting...")
                return best_model

            loss.backward()
            model.optimizer.step()

            loss_meter.update(loss.item())

        scheduler.step()

        logger.info("In epochs {} | Loss: {:5f}".format(
            epoch, loss_meter.avg
        ))
        
        if (epoch+1)%config.eval_epoch==0:
            print("Evaluate")
            valloss = eval_loss_rmtpp(model, val_loader, device)
            logger.info("Val Loss {:5f} ".format(valloss))
            if valloss < best_eval:
                best_eval = valloss
                best_model = copy.deepcopy(model)

    print("training done!")
    return best_model


def train(model, train_loader, val_loader, config, logger, device):
    
    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=150, gamma=0.2)
    best_eval = np.infty
    sll_meter = AverageMeter()
    tll_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    for epoch in trange(config.epochs):
        
        loss_total = 0
        model.train()
        
        for index, data in tenumerate(train_loader):
            st_x, st_y, _, _, _ = data
            
            

            model.optimizer.zero_grad()
            # sll: (batch_size, 1) = spatial log-likelihood
            loss, sll, tll = model.loss(st_x, st_y)

            if torch.isnan(loss):
                print("Numerical error, quiting...")
                return best_model
            _,w_i, b_i, inv_var = model(st_x.to(device))
            loss.backward()
            # Track gradients of the parameters in the model
            # Every 10 epochs
            # if epoch % 5 == 0:
            #     for name, param in model.named_parameters():
            #             # dont log background gradients
            #         if param.requires_grad and "background" not in name:
            #             experiment.log_metric(name + "_grad", param.grad.mean(), step=epoch)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            model.optimizer.step()
            loss_meter.update(loss.item())
            sll_meter.update(sll.mean())
            tll_meter.update(tll.mean())


        # Track loss
        experiment.log_metric("train_loss", loss_meter.avg, step=epoch)
        experiment.log_metric("train_sll", sll_meter.avg, step=epoch)
        # experiment.log_metric("train_tll", tll_meter.avg, step=epoch)

        # Track w_i,b_i and t_ti
        _, w_i, b_i, inv_var = model(st_x.to(device))
        w_i  = w_i.cpu().detach()
        b_i  = b_i.cpu().detach()
        inv_var = inv_var.cpu().detach()
        # experiment.log_metric("w_i", w_i.mean(), step=epoch)
        # experiment.log_metric("b_i", b_i.mean(), step=epoch)
        # experiment.log_metric("inv_var", inv_var.mean(), step=epoch)
        # # Track learning rate and gamma
        # experiment.log_metric("lr", model.optimizer.param_groups[0]['lr'], step=epoch)
        

        scheduler.step()

        logger.info("In epochs {} | "
                    "total loss: {:5f} | Space: {:5f} | Time: {:5f}".format(
            epoch, loss_meter.avg, sll_meter.avg , tll_meter.avg
        ))
        if (epoch+1)%config.eval_epoch==0:
            print("Evaluate")
            valloss, valspace, valtime = eval_loss(model, val_loader, device)
            logger.info("Val Loss {:5f} | Space: {:5f} | Time: {:5f}".format(valloss, valspace, valtime))
            if valloss < best_eval :
                best_eval = valloss
                best_model = copy.deepcopy(model)

    print("training done!")
    # Close comet experiment
    # experiment.end()
    return best_model


def mult_eval(models, n_eval, dataset, test_loader, config, device, scales, rmtpp=False):
    time_scale = np.log(scales[-1])
    space_scale = np.log(np.prod(scales[:2]))

    sll_list = []
    tll_list = []
    with torch.no_grad():
        for model in models:
            model.eval()
            for _ in trange(n_eval):
                if rmtpp:
                    tll = eval_loss_rmtpp(model, test_loader, device)
                    sll_list.append(0.0)
                    tll_list.append(-tll - time_scale)
                else:
                    _, sll, tll = eval_loss(model, test_loader, device)
                    sll_list.append(sll.item() - space_scale)
                    tll_list.append(tll.item() - time_scale)

    print("%.4f" % np.mean(sll_list), '±', "%.4f" % np.std(sll_list))
    print("%.4f" % np.mean(tll_list), '±', "%.4f" % np.std(tll_list))
    