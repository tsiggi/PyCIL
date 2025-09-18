import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

# Param for One Cold Cross Entropy
from utils.occe import OCCELoss
occe = OCCELoss()

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 150
lrate = 0.1
milestones = [60, 100, 130]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2
lamda = 3

print_test_acc_every_epoch = False

class LwF(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self._gamma = args["gamma"]
        self._kd_gamma = args.get("kd_gamma", None) 
        self._few_shot = args.get("few_shot", None)
        # _dual_head (Bool) param inherited from BaseLearner 
        
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            few_shot=self._few_shot
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, correct_head_2, total = 0, 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                _out = self._network(inputs)

                if self._network._dual_head :
                    logits, logits2 = _out[0]["logits"], _out[1]["logits"]
                    loss = F.cross_entropy(logits, targets) + self._gamma * occe(- logits2, targets)
                else :
                    logits = _out["logits"]
                    loss = F.cross_entropy(logits, targets) + self._gamma * occe(logits, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                # TODO : pred for dual head ?? store both outputs !!!
                if self._network._dual_head :
                    _, preds_of_head_2 = torch.max(-logits2, dim=1)
                    correct_head_2 += preds_of_head_2.eq(targets.expand_as(preds_of_head_2)).cpu().sum()
                    
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if self._network._dual_head :
                train_acc2 = np.around(tensor2numpy(correct_head_2) * 100 / total, decimals=2)

            if epoch % 5 == 0 or print_test_acc_every_epoch:
                if self._network._dual_head :
                    test_acc = self._compute_accuracy(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy: {}, Test_accy: {}".format(
                        self._cur_task,
                        epoch + 1,
                        epochs,
                        losses / len(train_loader),
                        {'ce_head': train_acc, 'occe_head': train_acc2},
                        test_acc
                    )
                else :
                    test_acc = self._compute_accuracy(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        init_epoch,
                        losses / len(train_loader),
                        train_acc,
                        test_acc,
                    )
            else:
                if self._network._dual_head :
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy: {}".format(
                        self._cur_task,
                        epoch + 1,
                        epochs,
                        losses / len(train_loader),
                        {'ce_head': train_acc, 'occe_head': train_acc2}
                    )
                else :
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        init_epoch,
                        losses / len(train_loader),
                        train_acc,
                    )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, correct_head_2, total = 0, 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                _out = self._network(inputs)

                if self._network._dual_head and self._kd_gamma is None :
                    logits, logits2 = _out[0]["logits"], _out[1]["logits"]

                    fake_targets = targets - self._known_classes
                    loss_clf = F.cross_entropy(
                        logits[:, self._known_classes :], fake_targets
                    )

                    occe_loss = occe(- logits2[:, self._known_classes :], fake_targets)

                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes],
                        self._old_network(inputs)[0]["logits"],
                        T,
                    )

                    loss = lamda * loss_kd + loss_clf + self._gamma * occe_loss 

                elif self._network._dual_head and self._kd_gamma is not None :
                    logits, logits2 = _out[0]["logits"], _out[1]["logits"]

                    fake_targets = targets - self._known_classes
                    loss_clf = F.cross_entropy(
                        logits[:, self._known_classes :], fake_targets
                    )

                    occe_loss = occe( - logits2[:, self._known_classes :], fake_targets)

                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes],
                        self._old_network(inputs)[0]["logits"],
                        T,
                    )
                    occe_kd_loss = _KD_loss(
                        - logits2[:, : self._known_classes],
                        - self._old_network(inputs)[1]["logits"],
                        T,
                    ) if self._kd_gamma is not None else 0.0

                    loss = lamda * loss_kd + self._kd_gamma* occe_kd_loss + loss_clf + self._gamma * occe_loss
                else :
                    logits = _out["logits"]

                    fake_targets = targets - self._known_classes
                    loss_clf = F.cross_entropy(
                        logits[:, self._known_classes :], fake_targets
                    )
                    occe_loss = occe(logits[:, self._known_classes :], fake_targets)
                
                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes],
                        self._old_network(inputs)["logits"],
                        T,
                    )

                    occe_kd_loss = self._kd_gamma*_KD_loss(
                        - logits[:, : self._known_classes],
                        - self._old_network(inputs)["logits"],
                        T,
                    ) if self._kd_gamma is not None else 0.0

                    loss = lamda * loss_kd + occe_kd_loss + loss_clf + self._gamma * occe_loss 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    if self._network._dual_head :
                        _, preds2 = torch.max(- logits2, dim=1)
                        correct_head_2 += preds2.eq(targets.expand_as(preds2)).cpu().sum()
                    total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if self._network._dual_head :
                train_acc2 = np.around(tensor2numpy(correct_head_2) * 100 / total, decimals=2)
            
            if epoch % 5 == 0 or print_test_acc_every_epoch:
                if self._network._dual_head :
                    test_acc = self._compute_accuracy(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy: {}, Test_accy: {}".format(
                        self._cur_task,
                        epoch + 1,
                        epochs,
                        losses / len(train_loader),
                        {'ce_head': train_acc, 'occe_head': train_acc2},
                        test_acc
                    )
                else : 
                    test_acc = self._compute_accuracy(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        epochs,
                        losses / len(train_loader),
                        train_acc,
                        test_acc,
                    )
            else:
                if self._network._dual_head :
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy: {}".format(
                        self._cur_task,
                        epoch + 1,
                        epochs,
                        losses / len(train_loader),
                        {'ce_head': train_acc, 'occe_head': train_acc2}
                    )
                else :
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        epochs,
                        losses / len(train_loader),
                        train_acc,
                    )
            prog_bar.set_description(info)
        logging.info(info)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
