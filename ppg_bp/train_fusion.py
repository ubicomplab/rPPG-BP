import os, sys, time, random, argparse
import numpy as np
import torch
import shutil
import json
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from model import M5, M5_fusion_conv
from dataset import UWBP_train_manual_demo, UWBP_val_manual_demo
from torch.utils.tensorboard import SummaryWriter

### python -m torch.distributed.launch --nproc_per_node=2 --master_port=45678 train_pre.py -c configs/default.yaml

class Trainer:
    def __init__(self, rank, world_size, config):
        self.config = config

        self.init_distributed(rank, world_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_datasets()
        # self.init_writer()
        self.train()
        self.cleanup()

    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        torch.cuda.set_device(self.rank % self.world_size)
        torch.distributed.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
    
    def cleanup(self):
        torch.distributed.destroy_process_group()

    def init_datasets(self):
        
        self.train_set = UWBP_train_manual_demo(self.config)
        self.val_set = UWBP_val_manual_demo(self.config)
        # self.train_set = UWBP_train_face_palm_skew(self.config)
        # self.val_set = UWBP_val_face_palm_skew(self.config)
        if self.config["dist"]:
            self.train_sampler = DistributedSampler(self.train_set, shuffle=True, drop_last=True)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, 
                batch_size=self.config["batch_size"]//len(self.config["gpus"]), 
                shuffle=False, 
                num_workers=self.config["num_workers"]//len(self.config["gpus"]), 
                drop_last=True, 
                pin_memory=True,
                sampler=self.train_sampler)
        else:
            self.train_loader = torch.utils.data.DataLoader(self.train_set, 
                batch_size=self.config["batch_size"], 
                shuffle=True, 
                num_workers=self.config["num_workers"], 
                drop_last=True, 
                pin_memory=True)
        
        self.val_loader = torch.utils.data.DataLoader(self.val_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    
    def output_log(self, log_str):
        log_file_path = os.path.join(self.config["output_dir"], "train_log.txt")
        with open(log_file_path, "a") as f:
            f.write(log_str)
    
    def init_model(self):
        if self.config["derivative_input"]:
            # self.model = M5(n_input=3, n_output=2)
            self.model = M5_fusion_conv(n_input=3, n_output=2)
        else:
            # self.model = M5_fusion(n_input=1, n_output=2)
            self.model = M5_fusion_conv(n_input=1, n_output=2)
            # self.model = FC_naive(n_input=2)

    def init_loss_and_optimizer(self):
        
        self.l1_criterion = torch.nn.L1Loss().to(self.device)
        self.l2_criterion = torch.nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=0)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config["scheduler_milestones"], gamma=self.config["scheduler_gamma"])


    def train_loss(self, batch, epoch, step):
        self.model.train()
        chunk = batch["ppg_chunk"].to(self.device).to(torch.float32)
        bp = batch["bp"].to(self.device).to(torch.float32)
        age = batch["age"].to(self.device).to(torch.float32)
        bmi = batch["bmi"].to(self.device).to(torch.float32)
        # gender = batch["gender"].to(self.device).to(torch.float32)

        bp_predict = self.model(chunk, age, bmi)
        # bp_predict = self.model(age, bmi)
        l1_loss = self.l2_criterion(bp_predict, bp)

        loss = l1_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.rank == 0:
            sys.stdout.write(f"\r[{epoch},{step}] loss {loss.item():.4f} l2 loss {l1_loss.item():.4f}")
            sys.stdout.flush()
        if "tb_writer" in self.config and self.config["tb_writer"] is not None:
            self.config["tb_writer"].add_scalar("loss/train", loss, step)
            self.config["tb_writer"].add_scalar("L2 loss/train", l1_loss, step)


    def train(self):

        epoch_start = 0
        step_start = 0

        # Random seed
        seed = self.config["manual_seed"]
        if seed is None:
            seed = random.randint(1, 10000)
        if self.rank == 0:
            print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # init model
        self.init_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        if self.config["dist"]:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)

        # loss and optimizer
        self.init_loss_and_optimizer()
        print("start learning rate: ", self.config["learning_rate"])
        
        output_dir = self.config["output_dir"]
        if not os.path.isdir(output_dir) and self.rank == 0:
            os.makedirs(output_dir)

        # DDP init model parameter
        init_snapshot_filename = "model_init.pth"
        init_snapshot_path = os.path.join(output_dir, init_snapshot_filename)
        if self.config["dist"] and not self.config["network"]["pretrained"]:
            if self.rank == 0:
                torch.save(self.model.state_dict(), init_snapshot_path)
        torch.distributed.barrier()

        # load pretrained weights if needed
        if self.config["network"]["pretrained"] is not None:
            snapshot_path = self.config["network"]["pretrained"]
        else:
            snapshot_path = init_snapshot_path
            print("Train from scratch")
        self.model.load_state_dict(torch.load(snapshot_path))
        # epoch_start, step_start, _ = load_checkpoint(snapshot_path, self.model, self.device)
        print("Loaded pretrained model.")
        
        # init
        if self.rank == 0:
            log_str = "*************************************************\n"
            log_str += f"Start training from step {step_start} at {time.strftime('%X %x')}\n"
            print(log_str)
            self.output_log(log_str)

            tb_dir=os.path.join(output_dir, "tb")
            self.config["tb_writer"] = SummaryWriter(log_dir=tb_dir, flush_secs=60)

        step_start = 0
        epochs_max = 6
        current_step = step_start
        for epoch in range(epoch_start, epochs_max):
            for i, batch in enumerate(self.train_loader):
                current_step += 1
                self.train_loss(batch, epoch, current_step)

                if current_step % self.config["steps_val"] == 0 and self.rank == 0:
                    snapshot_filename = "step_{}.pth".format(current_step)
                    snapshot_path = os.path.join(output_dir, snapshot_filename)
                    print("current learning rate: ", self.scheduler.get_lr())
                    self.valildate(batch, epoch, current_step)

                    # save_checkpoint(snapshot_path, model, optimizer, scheduler, epoch, current_step, float("inf"))
            self.scheduler.step()

    @torch.no_grad()
    def valildate(self, batch, epoch, step):
        self.model.eval()

        all_loss = []
        bp_sys_gt = []
        bp_sys_pred = []
        bp_dia_gt = []
        bp_dia_pred = []
        print("------------validating--------------")
        start_time = time.time()
        for i, data in enumerate(self.val_loader):
            temp_loss = list()
            tmp_by_sys_gt = list()
            tmp_bp_sys_pred = list()
            tmp_bp_dia_gt = list()
            tmp_bp_dia_pred = list()
            for j, batch in enumerate(data):
                chunks = batch["ppg_chunk"].to(self.device).to(torch.float32)
                bp = batch["bp"].to(self.device).to(torch.float32)
                age = batch["age"].to(self.device).to(torch.float32)
                bmi = batch["bmi"].to(self.device).to(torch.float32)
                # gender = batch["gender"].to(self.device).to(torch.float32)
                # bp_predict = self.model(age, bmi)
                bp_predict = self.model(chunks, age, bmi)
                # bp_predict = model(chunks)
                l2_loss = self.l2_criterion(bp_predict, bp).item()
                temp_loss.append(l2_loss)
                # tmp_by_sys_gt.append(torch.squeeze(bp).to("cpu").detach().numpy())
                # tmp_bp_sys_pred.append(torch.squeeze(bp_predict).to("cpu").detach().numpy())
                tmp_by_sys_gt.append(torch.squeeze(bp)[0].to("cpu").detach().numpy())
                tmp_bp_sys_pred.append(torch.squeeze(bp_predict)[0].to("cpu").detach().numpy())
                tmp_bp_dia_gt.append(torch.squeeze(bp)[1].to("cpu").detach().numpy())
                tmp_bp_dia_pred.append(torch.squeeze(bp_predict)[1].to("cpu").detach().numpy())

        # index = np.argmin(temp_loss)

            all_loss.append(np.mean(temp_loss))
            bp_sys_gt.append(np.mean(tmp_by_sys_gt))
            bp_sys_pred.append(np.mean(tmp_bp_sys_pred))
            bp_dia_gt.append(np.mean(tmp_bp_dia_gt))
            bp_dia_pred.append(np.mean(tmp_bp_dia_pred))

        end_time = time.time()

        avg_loss = np.mean(all_loss)
        # slope, intercept, sys_r_coef, p_values, se = scipy.stats.linregress(bp_sys_gt, bp_sys_pred)
        # slope, intercept, dia_r_coef, p_values, se = scipy.stats.linregress(bp_dia_gt, bp_dia_pred)
        bp_sys_matrix = torch.vstack((torch.tensor(bp_sys_gt), torch.tensor(bp_sys_pred)))
        sys_r_coef = torch.corrcoef(bp_sys_matrix)[0, 1]
        bp_dia_matrix = torch.vstack((torch.tensor(bp_dia_gt), torch.tensor(bp_dia_pred)))
        dia_r_coef = torch.corrcoef(bp_dia_matrix)[0, 1]

        duration = end_time - start_time
        print(f"\r[{epoch},{step}] val loss: {avg_loss:.4f}, sys pearson corr {sys_r_coef:.4f}, dia pearson corr {dia_r_coef:.4f}, duration {duration:.2f}")
        print("===============================================================")

        if "tb_writer" in self.config and self.config["tb_writer"] is not None:
            self.config["tb_writer"].add_scalar("Loss/val", avg_loss, step)
            self.config["tb_writer"].add_scalar("sys Pearson corr", sys_r_coef, step)
            self.config["tb_writer"].add_scalar("dia Pearson corr", dia_r_coef, step)

        # log
        output_dir = self.config["output_dir"]
        log_str = f"epoch {epoch} step {step} val loss: {avg_loss:.4f} sys pearson corr {sys_r_coef:.4f} dia pearson corr {dia_r_coef:.4f} \n"
        self.output_log(log_str)

        # save weights
        snapshot_filename = "step_{}.pth".format(step)
        snapshot_path = os.path.join(output_dir, snapshot_filename)
        torch.save(self.model.module.state_dict(), snapshot_path)

    # @torch.no_grad()
    # def valildate(self, batch, epoch, step):
    #     self.model.eval()

    #     all_loss = []
    #     bp_sys_gt = []
    #     bp_sys_pred = []
    #     bp_dia_gt = []
    #     bp_dia_pred = []
    #     print("------------validating--------------")
    #     start_time = time.time()
    #     for i, batch in enumerate(self.val_loader):

    #         chunk = batch["ppg_chunk"].to(self.device).to(torch.float32)
    #         bp = batch["bp"].to(self.device).to(torch.float32)
    #         age = batch["age"].to(self.device).to(torch.float32)
    #         bmi = batch["bmi"].to(self.device).to(torch.float32)
    #         gender = batch["gender"].to(self.device).to(torch.float32)

    #         bp_predict = self.model(chunk, age, bmi, gender)
    #         # bp_predict = self.model(age, bmi, gender)

    #         l1_loss = self.l2_criterion(bp_predict, bp).item()
    #         all_loss.append(l1_loss)
    #         # bp_sys_gt.append(torch.squeeze(bp))
    #         # bp_sys_pred.append(torch.squeeze(bp_predict))
    #         bp_sys_gt.append(torch.squeeze(bp)[0])
    #         bp_sys_pred.append(torch.squeeze(bp_predict)[0])
    #         bp_dia_gt.append(torch.squeeze(bp)[1])
    #         bp_dia_pred.append(torch.squeeze(bp_predict)[1])

    #         sys.stdout.write(f"\rval [{i + 1}/{len(self.val_loader)}] loss {l1_loss:.4f}")
    #         sys.stdout.flush()
    #     end_time = time.time()

    #     avg_loss = np.mean(all_loss)
    #     # slope, intercept, sys_r_coef, p_values, se = scipy.stats.linregress(bp_sys_gt, bp_sys_pred)
    #     # slope, intercept, dia_r_coef, p_values, se = scipy.stats.linregress(bp_dia_gt, bp_dia_pred)
    #     bp_sys_matrix = torch.vstack((torch.tensor(bp_sys_gt), torch.tensor(bp_sys_pred)))
    #     sys_r_coef = torch.corrcoef(bp_sys_matrix)[0, 1]
    #     bp_dia_matrix = torch.vstack((torch.tensor(bp_dia_gt), torch.tensor(bp_dia_pred)))
    #     dia_r_coef = torch.corrcoef(bp_dia_matrix)[0, 1]

    #     duration = end_time - start_time
    #     print(f"\r[{epoch},{step}] val loss: {avg_loss:.4f}, sys pearson corr {sys_r_coef:.4f}, dia pearson corr {dia_r_coef:.4f}, duration {duration:.2f}")
    #     print("===============================================================")

    #     if "tb_writer" in self.config and self.config["tb_writer"] is not None:
    #         self.config["tb_writer"].add_scalar("Loss/val", avg_loss, step)
    #         self.config["tb_writer"].add_scalar("sys Pearson corr", sys_r_coef, step)
    #         self.config["tb_writer"].add_scalar("dia Pearson corr", dia_r_coef, step)

    #     # log
    #     output_dir = self.config["output_dir"]
    #     log_str = f"epoch {epoch} step {step} val loss: {avg_loss:.4f} sys pearson corr {sys_r_coef:.4f} dia pearson corr {dia_r_coef:.4f} \n"
    #     self.output_log(log_str)

    #     # save weights
    #     snapshot_filename = "step_{}.pth".format(step)
    #     snapshot_path = os.path.join(output_dir, snapshot_filename)
    #     torch.save(self.model.module.state_dict(), snapshot_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument("-c", "--config", type=str, default="./config.json", help="config setting")
    opt = parser.parse_args()

    with open(opt.config, "r") as f:
        config = json.load(f)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '45678'
    gpu_list = ",".join(str(x) for x in config["gpus"])

    if not os.path.isdir(config["output_dir"]):
        os.makedirs(config["output_dir"])
    shutil.copyfile(opt.config, os.path.join(config["output_dir"], "test_config.json"))

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    world_size = torch.cuda.device_count()
    mp.spawn(
        Trainer,
        nprocs=world_size,
        args=(world_size, config,),
        join=True)




