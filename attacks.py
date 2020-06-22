import torch 
import torch.nn as nn
import torch.nn.functional as F
from perturbation_learning import cvae as CVAE
import math
from robustness.smoothing_core import Smooth

def CVAE_attack(X, y, cvae, model, max_dist=1, alpha=0.2, niters=10): 
    bs = X.size(0)
    with torch.no_grad(): 
        prior_params = cvae.prior(X)
        delta = torch.zeros_like(prior_params[0])
        delta.normal_()
        norm = delta.norm(p=2,dim=1)
        magnitude = max_dist*torch.rand(*norm.size()).to(norm.device)
        delta = (delta * (magnitude / norm).unsqueeze(1))

        # randomly init only if still correct at 0 perturbation
        # z = cvae.reparameterize(prior_params, eps=delta)
        # X_cvae = cvae.decode(X, z)
        # I = model(X_cvae).max(1)[1] == y

        # delta[I] = delta[I].normal_()
        # norm = delta.norm(p=2,dim=1)
        # magnitude = max_dist*torch.rand(*norm.size()).to(norm.device)
        # delta[I] = (delta * (magnitude / norm).unsqueeze(1))[I]
    # print("attacking")
    for i in range(niters):
        # print(f"iteration {i}")
        with torch.enable_grad():
            delta.requires_grad = True

            # generate perturbed image
            z = cvae.reparameterize(prior_params, eps=delta)
            X_cvae = cvae.decode(X, z)
            output = model(X_cvae)

            # done if all outputs are wrong
            I = output.max(1)[1] == y
            if I.ndim == 3: 
                I = I.view(I.size(0),-1).any(-1)
            if not I.any(): 
                # print(I.size())
                # print("breaking early")
                break



            # compute loss and backward
            loss = F.cross_entropy(output, y)
            loss.backward()

        with torch.no_grad(): 
            # take L2 gradient step
            g = delta.grad
            g = g / g.norm(p=2,dim=1).unsqueeze(1)
            delta[I] = (delta + alpha * g)[I]
            # project onto ball of radius max_dist
            delta[I] = delta.renorm(2,1,max_dist)[I]
        delta = delta.clone().detach()
    
    z = cvae.reparameterize(prior_params, eps=delta)
    X_cvae = cvae.decode(X, z).detach()
    return X_cvae, delta

def _CVAE_attack(config): 
    assert config.attack.type == 'cvae_attack'

    cvae = CVAE.models[config.attack.model.type](config.attack)
    cvae.to(config.device)

    d = torch.load(config.attack.checkpoint)
    cvae.load_state_dict(d['model_state_dict'])

    cvae.eval() 
    kwargs = {
        "max_dist": config.attack.max_dist, 
        "alpha": config.attack.alpha, 
        "niters": config.attack.niters
    }

    def forward(X,y,model): 
        return CVAE_attack(X, y, cvae, model, **kwargs)[0]

    return forward


def _CVAE_aug(config): 
    assert config.attack.type == 'cvae_aug'

    cvae = CVAE.models[config.attack.model.type](config.attack)
    cvae.to(config.device)

    d = torch.load(config.attack.checkpoint)
    cvae.load_state_dict(d['model_state_dict'])

    cvae.eval() 
    def forward(X,y,model): 
        with torch.no_grad(): 
            return cvae.sample(X)

    return forward

def _max_attack(config): 
    assert config.dataset.mode == "group"
    def forward(X,y,model): 
        with torch.no_grad(): 
            max_loss = None
            X_max = None
            for i in range(X.size(1)): 
                loss = F.cross_entropy(model(X[:,i,:,:,:]), y, reduction='none')
                loss = loss.view(loss.size(0),-1).mean(-1)
                if i == 0: 
                    X_max = X[:,i,:,:,:]
                    max_loss = loss
                else: 
                    I = loss > max_loss
                    X_max[I] = X[I,i,:,:,:]
                    max_loss = loss
            return X_max
    return forward


def _CVAE_gaussian(config): 
    assert config.attack.type == 'cvae_gaussian'
    # To draw samples at a radius R, note that 
    # the expected L2 norm of a random zero center gaussian 
    # with variance sigma^2*I is sqrt(N*sigma^2) where 
    # N is the number of dimensions. So to draw samples at 
    # an expected radius of R, we use sigma^2 = R^2/N
    cvae = CVAE.models[config.attack.model.type](config.attack)
    cvae.to(config.device)

    d = torch.load(config.attack.checkpoint)
    cvae.load_state_dict(d['model_state_dict'])
    cvae.eval() 

    N = config.attack.model.latent_dim
    if isinstance(N, list): 
        N = sum(N)
    sigma = config.attack.sigma
    # sigma = config.attack.radius/math.sqrt(N)
    def forward(X,y,model): 
        with torch.no_grad(): 
            eps = sigma*(X.new_empty(X.size(0), N).normal_())
            return cvae.sample(X, eps=eps)

    return forward

def _CVAE_certify(config): 
    assert config.attack.type == 'cvae_certify'
    cvae = CVAE.models[config.attack.model.type](config.attack)
    cvae.to(config.device)

    d = torch.load(config.attack.checkpoint)
    cvae.load_state_dict(d['model_state_dict'])
    cvae.eval() 

    N = config.attack.model.latent_dim
    if isinstance(N, list): 
        N = sum(N)
    sigma = config.attack.sigma
    n_classes = config.attack.n_classes
    selection_n0 = config.attack.selection_n0
    estimation_n = config.attack.estimation_n
    alpha = config.attack.alpha
    batch_size = config.eval.batch_size

    def forward(X,y,model): 
        smoothed_model = Smooth(model, cvae, n_classes, sigma)
        prediction, radius = smoothed_model.certify(X, selection_n0, estimation_n, alpha, batch_size)
        return prediction, radius
    return forward

def _CVAE_predict(config): 
    assert config.attack.type == 'cvae_predict'
    cvae = CVAE.models[config.attack.model.type](config.attack)
    cvae.to(config.device)

    d = torch.load(config.attack.checkpoint)
    cvae.load_state_dict(d['model_state_dict'])
    cvae.eval() 

    N = config.attack.model.latent_dim
    if isinstance(N, list): 
        N = sum(N)
    sigma = config.attack.sigma
    n_classes = config.attack.n_classes
    selection_n0 = config.attack.selection_n0
    estimation_n = config.attack.estimation_n
    alpha = config.attack.alpha
    batch_size = config.eval.batch_size

    def forward(X,y,model): 
        smoothed_model = Smooth(model, cvae, n_classes, sigma)
        prediction = smoothed_model.predict(X, selection_n0, alpha, batch_size)
        return prediction, torch.Tensor([0]).to(prediction.device)
    return forward


attacks = {
    "cvae_attack": _CVAE_attack, 
    "cvae_aug": _CVAE_aug, 
    "none": lambda config: (lambda X,y,model: X), 
    "max": _max_attack, 
    "cvae_gaussian": _CVAE_gaussian, 
    "cvae_certify": _CVAE_certify, 
    "cvae_predict": _CVAE_predict
}