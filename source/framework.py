import os

import piq
import torch
import torchvision
from iqa import IQA
from loss import TVLoss
from model import UnetTMO
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY


def save_image(im, p):
    base_dir = os.path.split(p)[0]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    torchvision.utils.save_image(im, p)


@MODEL_REGISTRY
class PSENet(LightningModule):
    def __init__(self, tv_w, gamma_lower, gamma_upper, number_refs, lr, afifi_evaluation=False):
        super().__init__()
        self.tv_w = tv_w
        self.gamma_lower = gamma_lower
        self.gamma_upper = gamma_upper
        self.number_refs = number_refs
        self.afifi_evaluation = afifi_evaluation
        self.lr = lr
        self.model = UnetTMO()
        self.mse = torch.nn.MSELoss()
        self.tv = TVLoss()
        self.iqa = IQA()
        self.saved_input = None
        self.saved_pseudo_gt = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=[0.9, 0.99])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss"}

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["total_loss"])

    def generate_pseudo_gt(self, im):
        bs, ch, h, w = im.shape
        underexposed_ranges = torch.linspace(0, self.gamma_upper, steps=self.number_refs)
        underexposed_gamma = torch.exp(
            torch.rand([bs, self.number_refs], device=im.device) * underexposed_ranges[None, :]
        )
        overrexposed_ranges = torch.linspace(self.gamma_lower, 0, steps=self.number_refs)
        overrexposed_gamma = torch.exp(
            torch.rand([bs, self.number_refs], device=im.device) * overrexposed_ranges[None, :]
        )
        gammas = torch.cat([underexposed_gamma, overrexposed_gamma], dim=1)
        # gammas: [bs, nref], im: [bs, ch, h, w] -> synthetic_references: [bs, nref, ch, h, w]
        synthetic_references = 1 - (1 - im[:, None]) ** gammas[:, :, None, None, None]
        previous_iter_output = self.model(im)[0].clone().detach()
        references = torch.cat([im[:, None], previous_iter_output[:, None], synthetic_references], dim=1)
        nref = references.shape[1]
        scores = self.iqa(references.view(bs * nref, ch, h, w))
        scores = scores.view(bs, nref, 1, h, w)
        print("DEBUG")
        max_idx = torch.argmax(scores, dim=1)
        print("DEBUG", max_idx.shape)
        max_idx = max_idx.repeat(1, ch, 1, 1)[:, None]
        pseudo_gt = torch.gather(references, 1, max_idx)
        return pseudo_gt.squeeze(1)

    def training_step(self, batch, batch_idx):
        # a hack to get the output from previous iterator
        # In nth interator, we use (n - 1)th batch instead of n-th batch to update model's weight. n-th batch will be used to generate a pseudo gt with current model's weigh and then is saved to use in (n + 1)th interator

        # saving n-th input and n-th pseudo gt
        nth_input = batch
        nth_pseudo_gt = self.generate_pseudo_gt(batch)
        if self.saved_input:
            # getting (n - 1)th input and (n - 1)-th pseudo gt -> calculate loss -> update model weight (handeled automatically by pytorch lightning)
            im = self.saved_input
            pred_im, pred_gamma = self.model(im)
            pseudo_gt = self.saved_pseudo_gt
            reconstruction_loss = self.mse(pred_im, pseudo_gt)
            tv_loss = self.tv(pred_gamma)
            loss = reconstruction_loss + tv_loss * self.tv_w

            # logging
            self.log(
                "train_loss/", {"reconstruction": reconstruction_loss, "tv": tv_loss}, on_epoch=True, on_step=False
            )
            self.log("total_loss", loss, on_epoch=True, on_step=False)
            if batch_idx == 0:
                visuals = [im, pseudo_gt, pred_im]
                visuals = torchvision.utils.make_grid([v[0] for v in visuals])
                self.logger.experiment.add_image("images", visuals, self.current_epoch)
        else:
            # skip updating model's weight at the first batch
            loss = None
            self.log("total_loss", 0, on_epoch=True, on_step=False)
        # saving n-th input and n-th pseudo gt
        self.saved_input = nth_input
        self.saved_pseudo_gt = nth_pseudo_gt
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            pred_im, pred_gamma = self.model(batch)
            self.logger.experiment.add_images("val_input", batch, self.current_epoch)
            self.logger.experiment.add_images("val_output", pred_im, self.current_epoch)

    def test_step(self, batch, batch_idx, test_idx=0):
        input_im, path = batch[0], batch[-1]
        pred_im, pred_gamma = self.model(input_im)
        for i in range(len(path)):
            save_image(pred_im[i], os.path.join(self.logger.log_dir, path[i]))

        if len(batch) == 3:
            gt = batch[1]
            psnr = piq.psnr(pred_im, gt)
            ssim = piq.ssim(pred_im, gt)
            self.log("psnr", psnr, on_step=False, on_epoch=True)
            self.log("ssim", ssim, on_step=False, on_epoch=True)
            if self.afifi_evaluation:
                assert len(path) == 1, "only support with batch size 1"
                if "N1." in path[0]:
                    self.log("psnr_under", psnr, on_step=False, on_epoch=True)
                    self.log("ssim_under", ssim, on_step=False, on_epoch=True)
                else:
                    self.log("psnr_over", psnr, on_step=False, on_epoch=True)
                    self.log("ssim_over", ssim, on_step=False, on_epoch=True)
