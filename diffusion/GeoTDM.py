"""
Code adapted from OpenAI guided diffusion repo:
https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
"""
import enum
import math

import numpy as np
import torch
from torch_geometric.nn import global_mean_pool, global_add_pool

from models.EGTN import EGTN
from .losses import normal_kl, gaussian_log_likelihood


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def sum_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)


class GeoTDM(object):
    """
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        denoise_network, beta_schedule_name, num_timesteps,
        model_mean_type, model_var_type, loss_type,
        device, rescale_timesteps=False, mode='cond',
    ):
        try:
            self.learn_ref_frame = denoise_network.learn_ref_frame
        except:
            self.learn_ref_frame = denoise_network.module.learn_ref_frame
        self.ref_frame_cache = None

        self.denoise_network: EGTN = denoise_network
        self.mode = mode
        self.device = device

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = get_named_beta_schedule(schedule_name=beta_schedule_name, num_diffusion_timesteps=num_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def project_to_subspace(self, x, batch):  # [BN, 3, T]
        x1 = global_mean_pool(x.mean(dim=-1), batch)  # [B, 3]
        x1 = x1[batch].unsqueeze(-1)  # [BN, 3, 1]
        return x - x1  # [BN, 3, T] in the subspace with dim (TN-1)D

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        x_ref = self.get_ref_frame(model_kwargs)

        if self.mode == 'cond':
            x = x + x_ref
        else:
            x = self.project_to_subspace(x, model_kwargs['batch'])

        model_output, _ = self.denoise_network(x=x, diffusion_t=self._scale_timesteps(t), **model_kwargs)

        if self.mode == 'cond':
            x = x - x_ref
        else:
            model_output = self.project_to_subspace(model_output, model_kwargs['batch'])

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = model_output
            else:
                pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t))
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self._scale_timesteps(t))

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(self, x, t, cond_fn=None, model_kwargs=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # Apply inpainting mask, similar to https://arxiv.org/abs/2201.09865
        # if model_kwargs is not None and 'cond_mask' in model_kwargs:
        #     assert 'x_given' in model_kwargs
        #     cond_mask = model_kwargs['cond_mask']  # [1, 1, T]
        #     x_given = model_kwargs['x_given']
        #     x_given_diffused = self.q_sample(x_start=x_given, t=t)
        #     x = cond_mask * x_given_diffused + (1 - cond_mask) * x

        out = self.p_mean_variance(x, t, model_kwargs=model_kwargs)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t)
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, shape, noise=None, cond_fn=None, progress=False, model_kwargs=None):
        """
        Generate samples from the model.

        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        self.clear_ref_frame()
        for sample in self.p_sample_loop_progressive(shape, noise=noise, cond_fn=cond_fn,
                                                     progress=progress, model_kwargs=model_kwargs):
            final = sample

        x_ref = self.get_ref_frame(model_kwargs)
        self.clear_ref_frame()
        if self.mode == 'cond':
            return final["sample"] + x_ref
        else:
            return final["sample"]

    def p_sample_loop_keeptraj(self, shape, keep_every=10, noise=None, cond_fn=None, progress=False, model_kwargs=None):
        """
        Generate samples from the model.

        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        traj = []
        step = 0  # The results collected would be [0, 100, 200,..., 999] if T=1000
        steps = []
        final = None
        self.clear_ref_frame()
        for sample in self.p_sample_loop_progressive(shape, noise=noise, cond_fn=cond_fn,
                                                     progress=progress, model_kwargs=model_kwargs):
            final = sample
            if step % keep_every == 0:
                traj.append(final["sample"])
                steps.append(step)
            step += 1

        # Always append the last frame
        traj.append(final["sample"])
        steps.append(step)
        print(steps)

        x_ref = self.get_ref_frame(model_kwargs)
        self.clear_ref_frame()
        if self.mode == 'cond':
            for i in range(len(traj)):
                traj[i] += x_ref
        return torch.stack(traj, dim=-1)  # [BN, 3, T, N_keep_step]

    def p_sample_loop_progressive(self, shape, noise=None, cond_fn=None, progress=False, model_kwargs=None):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=self.device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)  # Feed the same time step for all BN nodes
            with torch.no_grad():
                out = self.p_sample(img, t, cond_fn=cond_fn, model_kwargs=model_kwargs)
                yield out
                img = out["sample"]

    def get_ref_frame(self, model_kwargs=None):
        if self.mode == 'uncond':
            return None

        if self.ref_frame_cache is not None:
            return self.ref_frame_cache

        if not self.learn_ref_frame:  # Return the last given frame as ref frame
            cond_mask = model_kwargs['cond_mask']  # [1, 1, T]
            cond_mask = cond_mask.view(-1).bool()
            x_given = model_kwargs['x_given']
            x_cond = x_given[..., cond_mask]  # [BN, 3, T_c]
            batch = model_kwargs['batch']
            x_ref = global_mean_pool(x_cond.mean(dim=-1), batch)[batch].unsqueeze(-1)  # CoM of the cond trajectory
        else:  # Get the ref frame from the estimation network, per each future frame
            x_ref = self.denoise_network(diffusion_t=None, x=None, compute_ref=True, **model_kwargs)
            # [BN, 3, 1] or [BN, 3, T_f]

        self.ref_frame_cache = x_ref
        return x_ref

    def clear_ref_frame(self):
        self.ref_frame_cache = None

    def training_losses(self, x_start, t=None, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param x_start: the [N x C x ...] tensor of inputs. shape: [BN, 3, T]
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = torch.randn_like(x_start)
        if t is None:
            num_batch = int(model_kwargs['batch'].max() + 1)
            t = torch.randint(0, self.num_timesteps, size=(num_batch,)).to(x_start.device)  # [B,]
            t = t[model_kwargs['batch']]  # [BN,]

        # Construct reference frame
        self.clear_ref_frame()
        x_ref = self.get_ref_frame(model_kwargs)
        self.clear_ref_frame()

        if self.mode == 'cond':
            x_start = x_start - x_ref
        else:
            x_start = self.project_to_subspace(x_start, model_kwargs['batch'])
            noise = self.project_to_subspace(noise, model_kwargs['batch'])

        x_t = self.q_sample(x_start, t, noise=noise)  # [BN, 3, T_p]

        if self.mode == 'cond':
            x_t = x_t + x_ref

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output, _ = self.denoise_network(x=x_t, diffusion_t=self._scale_timesteps(t), **model_kwargs)

            if self.mode == 'uncond':
                model_output = self.project_to_subspace(model_output, model_kwargs['batch'])

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        nats.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in nats), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return sum_flat(kl_prior)

    def _vb_terms_bpd(
        self, x_start, x_t, t, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are nats. (I modified here from bits to nats.)
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            x_t, t, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )

        # if self.mode == 'uncond':
        #     # Adjust the likelihood since the dimension has changed
        #     adjust_term = 0.5 * (1 / (true_mean.size(2) * model_kwargs['num_nodes'][model_kwargs['batch']]))
        #     # 1 - (TN-1)D / TND = D / (TND) = 1 / TN
        #     kl = kl + adjust_term.view(-1, 1, 1)

        # kl = mean_flat(kl) / np.log(2.0)
        kl = sum_flat(kl)  # Here I compute total nll summed over all dimensions, instead of per dimension nll

        decoder_nll_constants, decoder_nll_term = gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"],
            batch=model_kwargs['batch'], num_nodes=model_kwargs['num_nodes'],
            subspace_dim_reduce=0 if self.mode == 'cond' else x_t.size(1)
        )

        assert decoder_nll_term.shape == x_start.shape

        decoder_nll = decoder_nll_constants + sum_flat(decoder_nll_term)
        # decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        # output = kl
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def calc_bpd_loop(self, x_start, model_kwargs=None, progress=False):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param x_start: the [N x C x ...] tensor of inputs.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        self.clear_ref_frame()

        if self.mode == 'cond':
            x_ref = self.get_ref_frame(model_kwargs)
            x_start = x_start - x_ref
        else:
            x_start = self.project_to_subspace(x_start, model_kwargs['batch'])

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        # Calc num_nodes
        batch = model_kwargs['batch']  # [BN]
        temp = torch.ones_like(batch)
        num_nodes = global_add_pool(temp, batch)  # [B]
        model_kwargs['num_nodes'] = num_nodes

        # for t in list(range(self.num_timesteps))[::-1]:
        for t in indices:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)

            if self.mode == 'uncond':
                noise = self.project_to_subspace(noise, model_kwargs['batch'])

            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd

        self.clear_ref_frame()
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def optimize(self, x_start, optimize_step, model_kwargs=None, noise=None, cond_fn=None):
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = torch.randn_like(x_start)

        t = torch.ones(x_start.shape[0]).to(x_start) * (optimize_step - 1)  # [BN]
        t = t.long()

        # Construct reference frame
        self.clear_ref_frame()
        x_ref = self.get_ref_frame(model_kwargs)
        self.clear_ref_frame()

        if self.mode == 'cond':
            x_start = x_start - x_ref
        else:
            x_start = self.project_to_subspace(x_start, model_kwargs['batch'])
            noise = self.project_to_subspace(noise, model_kwargs['batch'])

        x_t = self.q_sample(x_start, t, noise=noise)  # [BN, 3, T_p]

        # Run reverse denoising
        shape = x_t.shape
        img = x_t
        indices = list(range(optimize_step))[::-1]

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)  # Feed the same time step for all BN nodes
            with torch.no_grad():
                out = self.p_sample(img, t, cond_fn=cond_fn, model_kwargs=model_kwargs)
                img = out["sample"]

        return img + x_ref


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
