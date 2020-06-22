# Adapted from 
# https://github.com/locuslab/smoothing/blob/master/code/core.py

import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from torch.distributions.normal import Normal


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, cvae: torch.nn.Module, 
                 num_classes: int, sigma: float):
        """
        :param cvae: as specified in cvae.py
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.cvae = cvae
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        device = x.device
        self.cvae.eval()
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        # cAHat = counts_selection.argmax().item()
        cAHat = counts_selection.max(0)[1]
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        # nA = counts_estimation[cAHat].item()
        nA = counts_estimation.gather(0, cAHat.unsqueeze(0)).squeeze(0)

        # now all on CPU
        pABar = self._lower_confidence_bound(nA, n, alpha)

        std_normal = Normal(0,1)
        radius = self.sigma*std_normal.icdf(pABar)

        if cAHat.ndim == 0:         
            if pABar < 0.5:
                return torch.Tensor([Smooth.ABSTAIN]).long().to(device), torch.Tensor([0]).to(device)
            else:
                # radius = self.sigma * norm.ppf(pABar)
                return cAHat.to(device), radius.to(device)

        else: 
            I = pABar<0.5
            radius[I] = Smooth.ABSTAIN
            cAHat[I] = 0.0

            return cAHat.to(device), radius.to(device)

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        device = x.device
        self.cvae.eval()
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[-2:]
        #top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[1]].cpu().numpy()
        count2 = counts[top2[0]].cpu().numpy()
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return torch.Tensor([Smooth.ABSTAIN]).long().to(device)
        else:
            return top2[1].to(device)

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            x = x.unsqueeze(0)
            prior_params = self.cvae.prior(x)
            latent_dim = prior_params[0].size(1)
            counts = 0
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                # batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn(this_batch_size, latent_dim, device='cuda') * self.sigma
                # predictions = self.base_classifier(batch + noise).argmax(1)
                # Feed noise into CVAE instead of adding to input
                z = self.cvae.reparameterize(prior_params, eps=noise)
                batch = self.cvae.decode(x.expand((this_batch_size, -1, -1, -1)),z)
                predictions = self.base_classifier(batch).max(1)[1]
                if predictions.ndim == 3: 
                    counts += self._count_tensor(predictions, self.num_classes)
                else: 
                    counts += self._count_arr(predictions, self.num_classes)
            return counts

    def _count_tensor(self, tensor, length):
        counts = tensor.new_zeros(length, *tensor.size()[1:])
        labels = tensor.unique()
        for l in labels: 
            counts[l] = (tensor == l).sum(0)
        return counts

    def _count_arr(self, tensor, length: int) -> np.ndarray:
        counts = tensor.new_zeros(length, dtype=int)
        for idx in tensor:
            counts[idx] += 1
        return counts

    def _clopper_pearson(self, count, nobs, alpha): 
        """ Port of the clopper pearson test (beta branch) from 
        https://www.statsmodels.org/stable/_modules/statsmodels/stats/proportion.html#proportion_confint

        UNUSED BECAUSE PYTORCH DOESN'T SUPPORT INVERSE CDF FOR BETA
        """
        nobs = torch.ones_like(count)*nobs

        q_ = count * 1. / nobs
        alpha_2 = 0.5 * alpha

        ci_low = Beta(count, nobs - count + 1).icdf(alpha_2)
        ci_upp = Beta(count + 1, nobs - count).icdf(1-alpha_2)

        # ci_low = stats.beta.ppf(alpha_2, count, nobs - count + 1)
        # ci_upp = stats.beta.isf(alpha_2, count + 1, nobs - count)

        if ci_low.ndim > 0:
            ci_low[q_ == 0] = 0
            ci_upp[q_ == 1] = 1
        else:
            ci_low = ci_low if (q_ != 0) else 0
            ci_upp = ci_upp if (q_ != 1) else 1

        return ci_low, ci_upp

    def _lower_confidence_bound(self, NA, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        if NA.ndim == 2: 
            output = torch.zeros(*NA.size())
            for i in range(NA.size(0)): 
                for j in range(NA.size(1)): 
                    output[i,j] = proportion_confint(NA[i][j].item(), N, alpha=2 * alpha, method="beta")[0]
            return output.cuda()
        else: 
            #print(NA.item(), N, alpha) 
            bound = proportion_confint(NA.item(), N, alpha=2 * alpha, method="beta")[0]
            output = torch.Tensor([bound])
            return output.cuda()

        # return self._clopper_pearson(NA, N, 2 * alpha)[0]