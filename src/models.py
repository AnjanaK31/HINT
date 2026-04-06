import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .networks import HINT, Discriminator, MobileNetV2
from .networks import HINT, Discriminator, MobileNetV2
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss



class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else: 
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'], strict=False)
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            try:
                self.discriminator.load_state_dict(data['discriminator'])
            except RuntimeError as e:
                print(f"WARNING: Could not load discriminator due to size mismatch (likely 4-channel to 5-channel transition). Training discriminator from scratch. Error: {e}")

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)



class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        generator = HINT()
        discriminator = Discriminator(in_channels=5, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )


    def process(self, images, masks, landmark_map=None):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        if landmark_map is None:
            landmark_map = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3]).to(images.device)

        # process outputs
        outputs_img = self(images, masks, landmark_map)

        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs_img.detach()

        dis_real, _ = self.discriminator(torch.cat((dis_input_real, masks, landmark_map), dim=1))                  
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake, masks, landmark_map), dim=1))                  

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs_img
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, masks, landmark_map), dim=1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        gen_l1_loss = self.l1_loss(outputs_img, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs_img, images) * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs_img * masks, images * masks) * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # ---> THE SYMMETRY LOSS (LAFIN Inspired) <---
        gen_sym_loss = 0
        if self.config.USE_LANDMARKS and landmarks is not None:
            # Simple global flip as baseline
            flipped_outputs = torch.flip(outputs_img, dims=[3])
            sym_weight = getattr(self.config, 'SYMMETRY_LOSS_WEIGHT', 10.0)
            
            # The core idea: Local regions near symmetric landmarks should be similar
            # In LAFIN, this is often done by aligning or weighting the loss
            # Here we implement it as a combination of global flip and potential specific checks
            gen_sym_loss = self.l1_loss(outputs_img, flipped_outputs) * sym_weight
        else:
            # Fallback for no landmarks
            flipped_outputs = torch.flip(outputs_img, dims=[3])
            sym_weight = getattr(self.config, 'SYMMETRY_LOSS_WEIGHT', 5.0)
            gen_sym_loss = self.l1_loss(outputs_img, flipped_outputs) * sym_weight

        gen_loss += gen_sym_loss
        # ---------------------------

        # create logs
        logs = [
            ("gLoss", gen_loss.item()),
            ("dLoss", dis_loss.item()),
            ("symLoss", gen_sym_loss.item())
        ]

        return outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, gen_sym_loss


    def forward(self, images, masks, landmark_map=None):
        images_masked = (images * (1 - masks).float()) + masks

        scaled_masks_tiny = F.interpolate(masks, size=[int(masks.shape[2] / 8), int(masks.shape[3] / 8)],
                                     mode='nearest')        
        
        scaled_masks_quarter = F.interpolate(masks, size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
                                     mode='nearest')
        scaled_masks_half = F.interpolate(masks, size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
                                     mode='nearest')

        outputs_img = self.generator(inputs,masks,scaled_masks_half,scaled_masks_quarter,scaled_masks_tiny, landmark_map)            
        return outputs_img

    def backward(self, gen_loss = None, dis_loss = None, scaler = None):
        optimizer_dis = self.dis_optimizer
        optimizer_gen = self.gen_optimizer

        if scaler:
            # Scaled Discriminator backward
            optimizer_dis.zero_grad()
            scaler.scale(dis_loss).backward(retain_graph=True)
            scaler.step(optimizer_dis)

            # Scaled Generator backward
            optimizer_gen.zero_grad()
            scaler.scale(gen_loss).backward()
            scaler.step(optimizer_gen)

            scaler.update()
        else:
            dis_loss.backward(retain_graph= True)
            self.dis_optimizer.step()

            gen_loss.backward()
            self.gen_optimizer.step()

    def backward_joint(self, gen_loss = None, dis_loss = None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()



def abs_smooth(x):
    absx = torch.abs(x)
    minx = torch.min(absx,other=torch.ones(absx.shape).cuda())
    r = 0.5 *((absx-1)*minx + absx)
    return r

def loss_landmark_abs(y_true, y_pred):
    loss = torch.mean(abs_smooth(y_pred - y_true))
    return loss

def loss_landmark(landmark_true, landmark_pred, points_num=68):
    landmark_loss = torch.norm((landmark_true-landmark_pred).reshape(-1,points_num*2),2,dim=1,keepdim=True)

    return torch.mean(landmark_loss)

class LandmarkDetectorModel(nn.Module):
    def __init__(self, config):
        super(LandmarkDetectorModel, self).__init__()
        self.mbnet = MobileNetV2(points_num=config.LANDMARK_POINTS)
        self.name = 'landmark_detector'
        self.iteration = 0
        self.config = config

        self.landmark_weights_path = os.path.join(config.PATH, self.name + '.pth')

        if len(config.GPU) > 1:
            self.mbnet = nn.DataParallel(self.mbnet, config.GPU)

        self.optimizer = optim.Adam(
            params=self.mbnet.parameters(),
            lr=self.config.LR,
            weight_decay=0.000001
        )


    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'detector': self.mbnet.state_dict()
        }, self.landmark_weights_path)

    def load(self):
        if os.path.exists(self.landmark_weights_path):
            print('Loading landmark detector...')

            if torch.cuda.is_available():
                data = torch.load(self.landmark_weights_path)
            else:
                data = torch.load(self.landmark_weights_path, map_location=lambda storage, loc: storage)

            self.mbnet.load_state_dict(data['detector'])
            self.iteration = data['iteration']
            print('Loading landmark detector complete!')

    def forward(self, images, masks):
        images_masked = images* (1 - masks).float() + masks

        landmark_gen = self.mbnet(images_masked)
        landmark_gen *= self.config.INPUT_SIZE

        return landmark_gen

    def process(self, images, masks, landmark_gt):
        self.iteration += 1
        self.optimizer.zero_grad()

        images_masked = images*(1-masks)+masks
        landmark_gen = self(images_masked, masks)
        landmark_gen = landmark_gen.reshape((-1, self.config.LANDMARK_POINTS, 2))
        loss = loss_landmark(landmark_gt.float(),landmark_gen, points_num=self.config.LANDMARK_POINTS)

        logs = [("loss", loss.item())]
        return landmark_gen, loss, logs

    def process_aug(self, images, masks, landmark_gt):
        self.optimizer.zero_grad()
        images_masked = images*(1-masks)+masks
        landmark_gen = self(images_masked, masks)
        landmark_gen = landmark_gen.reshape(-1,self.config.LANDMARK_POINTS,2)
        loss = loss_landmark(landmark_gt.float(),landmark_gen, points_num=self.config.LANDMARK_POINTS)

        logs = [("loss_aug", loss.item())]

        return landmark_gen, loss, logs



    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
