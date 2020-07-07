import os
import numpy as np
import torch


class ModelEvaluator(object):
    def __init__(self, model, FLAGS):
        self.model = model
        self.FLAGS = FLAGS

    def vae_eval(self, data):
        model = self.model.eval()
        vae = model

        if self.FLAGS.cuda:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        vae = vae.to(dev)

        start = data['start'][:100]
        start_mean = np.mean(start, axis=0, keepdims=True)
        start_normalized = (start - start_mean)
        start_mean = np.tile(start_mean, (start.shape[0], 1))

        start = np.concatenate([start_normalized, start_mean], axis=1)

        transformation = data['object_mask_down'][:100, None]
        kd_idx = np.arange(100, dtype=np.int64)

        start = torch.from_numpy(start)
        transformation = torch.from_numpy(transformation)
        kd_idx = torch.from_numpy(kd_idx)

        joint_keypoint = torch.cat([start.float(), transformation.float()], dim=1)
        joint_keypoint = joint_keypoint[None, :, :]
        kd_idx = kd_idx[None, :]

        joint_keypoint = joint_keypoint.float().to(dev)
        kd_idx = kd_idx.long().to(dev)

        z = torch.randn(1, self.FLAGS.latent_dimension).to(dev)
        geom_embed = model.geom_encoder(joint_keypoint)
        repeat = kd_idx.size(1)
        ex_wt = model.exist_wt(geom_embed)
        kd_idx = kd_idx[:, :, None].repeat(1, 1, geom_embed.size(2))[:, :repeat]
        geom_embed = torch.gather(geom_embed, 1, kd_idx)

        size = geom_embed.size()
        z = z[:, None, :].repeat(1, size[1], 1)
        z = (z * geom_embed)

        inp = torch.cat((z, geom_embed), dim=-1)
        recon_mu = model.decoder(inp)
        output_r, output_l = recon_mu

        output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()
        output_joint = np.concatenate([output_r, output_l], axis=2)
        ex_wt = ex_wt.detach().cpu().numpy().squeeze()

        sort_idx = np.argsort(ex_wt)[None, :]
        predictions = []

        for i in range(output_joint.shape[0]):
            for j in range(output_joint.shape[1]):
                j = sort_idx[i, j]
                pred_info = output_joint[i, j]
                predictions.append(pred_info.tolist())
        predictions = np.asarray(predictions).squeeze()
        return predictions, sort_idx

    def vae_eval_goal(self, data):
        model = self.model.eval()
        vae = model

        if self.FLAGS.cuda:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        vae = vae.to(dev)

        with torch.no_grad():
            object_mask_down = data['object_mask_down'][:100]
            start = data['start'][:100]
            start_mean = np.mean(start, axis=0, keepdims=True)
            start_normalized = (start - start_mean)
            start_mean = np.tile(start_mean, (start.shape[0], 1))

            start = np.concatenate([start_normalized, start_mean], axis=1)

            start = torch.from_numpy(start)[None, :, :]
            object_mask_down = torch.from_numpy(object_mask_down)[None, :]

            object_mask_down = object_mask_down[:, :, None].float()
            start = start.float()
            joint_keypoint = torch.cat([start, object_mask_down], dim=2)

            joint_keypoint = joint_keypoint.float().to(dev)
            start = start.to(dev)
            object_mask_down = object_mask_down.float().to(dev)

            masks = []
            # Sample 10 different keypoints from the model
            for repeat in range(10):
                z = torch.randn(start.size(0), self.FLAGS.latent_dimension).to(dev)
                pred_mask = model.decode(z, start).detach().cpu().numpy()
                masks.append(pred_mask)
        return np.asarray(masks).squeeze()

    def vqvae_eval_goal(self, data):
        model = self.model.eval()
        vae = model

        if self.FLAGS.cuda:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        vae = vae.to(dev)

        with torch.no_grad():
            # object_mask_down = data['object_mask_down'][:100]
            start = data['start'][:100]
            start_mean = np.mean(start, axis=0, keepdims=True)
            start_normalized = (start - start_mean)
            start_mean = np.tile(start_mean, (start.shape[0], 1))

            start = np.concatenate([start_normalized, start_mean], axis=1)

            start = torch.from_numpy(start)[None, :, :]
            # object_mask_down = torch.from_numpy(object_mask_down)[None, :]

            # object_mask_down = object_mask_down[:, :, None].float()
            start = start.float()
            # joint_keypoint = torch.cat([start, object_mask_down], dim=2)

            # joint_keypoint = joint_keypoint.float().to(dev)
            start = start.to(dev)
            # object_mask_down = object_mask_down.float().to(dev)

            masks = []
            # Sample 10 different keypoints from the model
            for repeat in range(10):
                z = torch.randn(start.size(0), self.FLAGS.latent_dimension).to(dev)
                pred_mask = model.decode(z, start)[0].detach().cpu().numpy()
                masks.append(pred_mask)
        return np.asarray(masks).squeeze()

    def vae_eval_joint(self, data, pointnet=False, pulling=False):
        model = self.model.eval()
        vae = model

        if self.FLAGS.cuda:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        vae = vae.to(dev)

        start = data['start'][:100]

        # if pulling:
        #     start_mean = np.array([0.0, 0.0, 0.0])
        #     centroid = np.array([0.0, 0.0, 0.0])
        # else:
        #     start_mean = np.mean(start, axis=0, keepdims=True)
        start_mean = np.mean(start, axis=0, keepdims=True)
        start_normalized = (start - start_mean)
        start_mean = np.tile(start_mean, (start.shape[0], 1))

        start = np.concatenate([start_normalized, start_mean], axis=1)
        kd_idx = np.arange(100, dtype=np.int64)

        start = torch.from_numpy(start)
        kd_idx = torch.from_numpy(kd_idx)

        joint_keypoint = start
        joint_keypoint = joint_keypoint[None, :, :]
        kd_idx = kd_idx[None, :]

        joint_keypoint = joint_keypoint.float().to(dev)
        kd_idx = kd_idx.long().to(dev)

        palm_predictions = []
        mask_predictions = []
        trans_predictions = []


        for repeat in range(10):
            palm_repeat = []
            z = torch.randn(1, self.FLAGS.latent_dimension).to(dev)
            recon_mu, ex_wt = model.decode(z, joint_keypoint)
            if len(recon_mu) == 4:
                output_r, output_l, pred_mask, pred_trans = recon_mu
                trans_predictions.append(pred_trans.detach().cpu().numpy())
            elif len(recon_mu) == 5:
                output_r, output_l, pred_mask, pred_trans, pred_transform = recon_mu
                trans_predictions.append(pred_transform.detach().cpu().numpy())
            mask_predictions.append(pred_mask.detach().cpu().numpy())
            output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()

            if pointnet:
                output_joint = np.concatenate([output_r, output_l], axis=1)
                palm_repeat.append(output_joint)
            else:
                output_joint = np.concatenate([output_r, output_l], axis=2)
                ex_wt = ex_wt.detach().cpu().numpy().squeeze()
                sort_idx = np.argsort(ex_wt)[None, :]

                for i in range(output_joint.shape[0]):
                    for j in range(output_joint.shape[1]):
                        j = sort_idx[i, j]
                        pred_info = output_joint[i, j]
                        palm_repeat.append(pred_info.tolist())
            palm_predictions.append(np.asarray(palm_repeat))
        palm_predictions = np.asarray(palm_predictions).squeeze()
        mask_predictions = np.asarray(mask_predictions).squeeze()
        trans_predictions = np.asarray(trans_predictions).squeeze()

        return palm_predictions, mask_predictions, trans_predictions

    def vae_eval_joint_T(self, data, pointnet=False):
        model = self.model.eval()
        vae = model

        if self.FLAGS.cuda:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        vae = vae.to(dev)

        start = data['start'][:100]
        start_mean = np.mean(start, axis=0, keepdims=True)
        start_normalized = (start - start_mean)
        start_mean = np.tile(start_mean, (start.shape[0], 1))

        start = np.concatenate([start_normalized, start_mean], axis=1)
        kd_idx = np.arange(100, dtype=np.int64)

        start = torch.from_numpy(start)
        kd_idx = torch.from_numpy(kd_idx)

        joint_keypoint = start
        joint_keypoint = joint_keypoint[None, :, :]
        kd_idx = kd_idx[None, :]

        joint_keypoint = joint_keypoint.float().to(dev)
        kd_idx = kd_idx.long().to(dev)

        palm_predictions = []
        # mask_predictions = []
        trans_predictions = []

        for repeat in range(10):
            palm_repeat = []
            z = torch.randn(1, self.FLAGS.latent_dimension).to(dev)
            recon_mu, ex_wt = model.decode(z, joint_keypoint)
            # output_r, output_l, pred_mask, pred_trans = recon_mu
            output_r, output_l, pred_trans = recon_mu
            # mask_predictions.append(pred_mask.detach().cpu().numpy())
            trans_predictions.append(pred_trans.detach().cpu().numpy())

            output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()

            if pointnet:
                output_joint = np.concatenate([output_r, output_l], axis=1)
                palm_repeat.append(output_joint)
            else:
                output_joint = np.concatenate([output_r, output_l], axis=2)
                ex_wt = ex_wt.detach().cpu().numpy().squeeze()
                sort_idx = np.argsort(ex_wt)[None, :]

                for i in range(output_joint.shape[0]):
                    for j in range(output_joint.shape[1]):
                        j = sort_idx[i, j]
                        pred_info = output_joint[i, j]
                        palm_repeat.append(pred_info.tolist())
            palm_predictions.append(np.asarray(palm_repeat))
        palm_predictions = np.asarray(palm_predictions).squeeze()
        # mask_predictions = np.asarray(mask_predictions).squeeze()
        trans_predictions = np.asarray(trans_predictions).squeeze()


        return palm_predictions, trans_predictions        


    def optimize_latent(self, data):

        # obj_frame = data['']
        if self.FLAGS.cuda:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")


        start = data['start'][:100]
        start_mean = np.mean(start, axis=0, keepdims=True)
        start_normalized = (start - start_mean)
        start_mean = np.tile(start_mean, (start.shape[0], 1))

        start = np.concatenate([start_normalized, start_mean], axis=1)
        kd_idx = np.arange(100, dtype=np.int64)

        start = torch.from_numpy(start).float().to(dev)[None, :, :]

        z = torch.randn(1, self.FLAGS.latent_dimension).to(dev)
        # print(start.size())
        # print(z.size())
        # target_batch_left, target_batch_right = torch.chunk(obj_frame, 2, dim=1)

        with torch.enable_grad():
            for i in range(50):
                # print(i)
                # z.requires_grad_(True)
                z.requires_grad = True
                recon_mu, ex_wt = self.model.decode(z, start)
                output_r, output_l, pred_mask, pred_trans, pred_transform = recon_mu

                object_mask_down = (pred_mask > 0.95).float()
                transformation = pred_transform
                translation = pred_trans
                obj_frame = torch.cat([output_l, output_r], dim=-1)

                # Generate x from predicted output_r and output_l
                joint_keypoint = torch.cat(
                    [start, object_mask_down, obj_frame,
                    transformation[:, None, :].repeat(1, start.size(1), 1).float(),
                    translation[:, None, :].repeat(1, start.size(1), 1).float()], dim=2)

                z_new, _, _ = self.model.encode(joint_keypoint)
                dist = torch.norm(z - z_new, p=2, dim=-1).sum()
                # print("dist: ", dist)
                # right_norm = torch.norm(target_batch_right.view(-1, 100, 7) - output_r, p=2, dim=-1)
                # left_norm = torch.norm(target_batch_left.view(-1, 100, 7) - output_l, p=2, dim=-1)
                # tot_norm = right_norm.sum() + left_norm.sum()
                z_grad = torch.autograd.grad([dist], [z])[0]

                # print(i, "distance: ", (right_norm + left_norm).min(dim=1)[0].mean())

                z = z - 0.5 * z_grad
                z = z.detach()

        # _, min_idx = (right_norm + left_norm).min(dim=1)
        kd_idx = None
        # kd_idx = torch.gather(kd_idx, 1, min_idx[:, None])
        # min_idx = min_idx[:, None, None].repeat(1, 1, FLAGS.latent_dimension)
        # z = torch.gather(z, 1, min_idx)

        return z, kd_idx


    def vae_eval_joint_optimize(self, data, pointnet=False, pulling=False):
        model = self.model.eval()
        vae = model

        if self.FLAGS.cuda:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        vae = vae.to(dev)

        start = data['start'][:100]

        # if pulling:
        #     start_mean = np.array([0.0, 0.0, 0.0])
        #     centroid = np.array([0.0, 0.0, 0.0])
        # else:
        #     start_mean = np.mean(start, axis=0, keepdims=True)
        start_mean = np.mean(start, axis=0, keepdims=True)
        start_normalized = (start - start_mean)
        start_mean = np.tile(start_mean, (start.shape[0], 1))

        start = np.concatenate([start_normalized, start_mean], axis=1)
        kd_idx = np.arange(100, dtype=np.int64)

        start = torch.from_numpy(start)
        kd_idx = torch.from_numpy(kd_idx)

        joint_keypoint = start
        joint_keypoint = joint_keypoint[None, :, :]
        kd_idx = kd_idx[None, :]

        joint_keypoint = joint_keypoint.float().to(dev)
        kd_idx = kd_idx.long().to(dev)

        palm_predictions = []
        mask_predictions = []
        trans_predictions = []


        for repeat in range(3):
            palm_repeat = []
            # z = torch.randn(1, self.FLAGS.latent_dimension).to(dev)
            z = self.optimize_latent(data)[0]
            recon_mu, ex_wt = model.decode(z, joint_keypoint)
            if len(recon_mu) == 4:
                output_r, output_l, pred_mask, pred_trans = recon_mu
                trans_predictions.append(pred_trans.detach().cpu().numpy())
            elif len(recon_mu) == 5:
                output_r, output_l, pred_mask, pred_trans, pred_transform = recon_mu
                trans_predictions.append(pred_transform.detach().cpu().numpy())
            mask_predictions.append(pred_mask.detach().cpu().numpy())
            output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()

            if pointnet:
                output_joint = np.concatenate([output_r, output_l], axis=1)
                palm_repeat.append(output_joint)
            else:
                output_joint = np.concatenate([output_r, output_l], axis=2)
                ex_wt = ex_wt.detach().cpu().numpy().squeeze()
                sort_idx = np.argsort(ex_wt)[None, :]

                for i in range(output_joint.shape[0]):
                    for j in range(output_joint.shape[1]):
                        j = sort_idx[i, j]
                        pred_info = output_joint[i, j]
                        palm_repeat.append(pred_info.tolist())
            palm_predictions.append(np.asarray(palm_repeat))
        palm_predictions = np.asarray(palm_predictions).squeeze()
        mask_predictions = np.asarray(mask_predictions).squeeze()
        trans_predictions = np.asarray(trans_predictions).squeeze()

        return palm_predictions, mask_predictions, trans_predictions        