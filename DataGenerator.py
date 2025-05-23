import torch
import numpy as np

# Generates a tensor of Brownian motion increments with shape (sequence_len, batch_size, dim).
def gen_dB(sequence_len, batch_size, dim, T):
    return torch.randn(sequence_len,batch_size,dim)*np.sqrt(T/sequence_len)

#Generates a tensor of Brownian motion up to time t with shape (sequence_len, batch_size, dim).
def gen_B_t(batch_size, dim, t):
    return torch.randn(batch_size,dim)*np.sqrt(t)

################################################################
# Methods that generates a tensor of jumps with shape (sequence_len, batch_size x dim),
# given lists of parameters for distribution

# This returns log-normal jumps corresponding to Merton model.
def gen_jumps_ln(sequence_len, batch_size, mus, sigmas):
    mus_t = torch.tensor(mus).float().unsqueeze(0)
    mus_t = torch.repeat_interleave(mus_t,batch_size,0)
    mus_t = mus_t.view(-1).unsqueeze(0)
    mus_t = torch.repeat_interleave(mus_t,sequence_len,0)

    sigmas_t = torch.tensor(sigmas).float().unsqueeze(0)
    sigmas_t = torch.repeat_interleave(sigmas_t, batch_size, 0)
    sigmas_t = sigmas_t.view(-1).unsqueeze(0)
    sigmas_t = torch.repeat_interleave(sigmas_t, sequence_len, 0)

    return torch.exp(torch.normal(mus_t, sigmas_t)) - 1

#This returns jumps realized from a mixture of log-normal distributions
def gen_jumps_mixture(sequence_len, batch_size, mus, sigmas, rates):

    #Auxiliary function that mixes two distributions
    def mix_two(p, x, y):
        mask = torch.bernoulli(torch.full(x.shape, p)).int()
        reverse_mask = torch.ones(x.shape).int() - mask
        return x * mask + y * reverse_mask

    mu_t = torch.ones(sequence_len, batch_size)*mus[0]
    sigma_t = torch.ones(sequence_len, batch_size)*sigmas[0]

    jumps = torch.exp(torch.normal(mu_t, sigma_t)) - 1

    for i in range(len(rates)-1):
        p = sum(rates[:i+1])/sum(rates[:i+2])
        mu_i = torch.ones(sequence_len, batch_size) * mus[i+1]
        sigma_i = torch.ones(sequence_len, batch_size) * sigmas[i+1]
        jumps_i = torch.exp(torch.normal(mu_i, sigma_i)) - 1
        jumps = mix_two(p,jumps, jumps_i)

    return jumps

# Returns jumps e^y -1 for y double exponential distributed, corresponding to Kou's model.
def gen_jumps_de(sequence_len, batch_size, etas1, etas2,ps):
    etas1_t = torch.tensor(etas1).float().unsqueeze(0)
    etas1_t = torch.repeat_interleave(etas1_t,batch_size,0)
    etas1_t = etas1_t.view(-1).unsqueeze(0)
    etas1_t = torch.repeat_interleave(etas1_t,sequence_len,0)

    etas2_t = torch.tensor(etas2).float().unsqueeze(0)
    etas2_t = torch.repeat_interleave(etas2_t, batch_size, 0)
    etas2_t = etas2_t.view(-1).unsqueeze(0)
    etas2_t = torch.repeat_interleave(etas2_t, sequence_len, 0)

    ps_t = torch.tensor(ps).float().unsqueeze(0)
    ps_t = torch.repeat_interleave(ps_t, batch_size, 0)
    ps_t = ps_t.view(-1).unsqueeze(0)
    ps_t = torch.repeat_interleave(ps_t, sequence_len, 0)

    mask = torch.bernoulli(ps_t)

    exp1 = torch.distributions.exponential.Exponential(etas1_t)
    exp2 = torch.distributions.exponential.Exponential(etas2_t)

    return torch.exp(mask*exp1.sample() - (1-mask)*exp2.sample()) - 1

##########################################

# Generates a boolean tensor of jump times. Used later with gen_compound_poisson
def gen_jump_times(sequence_len, batch_size, rates, T):

    dim = len(rates)
    rates_t = torch.tensor(rates).float().unsqueeze(0)
    rates_t = torch.repeat_interleave(rates_t, batch_size, 0)

    N = torch.poisson(rates_t*T)
    N = N.view(-1)

    M = N.reshape(batch_size, dim)

    max_length = int(N.max())
    bs2 = int(N.size()[0])

    tj = torch.randint(sequence_len, (bs2, max_length))
    range_tensor = torch.arange(max_length)
    bool_tensor = range_tensor >= N.view(-1, 1)
    tj[bool_tensor] = sequence_len

    return tj

# Generates a tensor of incrementes for a compensated compound Poisson process with jump realization given by parameter
# "jumps" and jump times given by parameter "jump_times". The output tensor is of shape (sequence_len, batch_size, dim).
def gen_compound_poisson(sequence_len, batch_size, dim, jumps, jump_times):
    jumps = torch.cat((jumps, torch.ones(1, batch_size * dim)), dim=0)
    dP = torch.zeros(sequence_len + 1, batch_size * dim).scatter_(0, jump_times.transpose(0, 1), jumps, reduce="add")[:-1, :]

    return dP.reshape(sequence_len, batch_size, dim)

#Generates a compensator tensor under Levy measures of shape (dim)
#Compound Poisson process with log-normal jump distribution
def gen_compensator_ln(rates,mus,sigmas):
    rates = torch.tensor(rates).float()
    mus = torch.tensor(mus).float()
    sigmas = torch.tensor(sigmas).float()
    return rates * (torch.exp(mus + 0.5*sigmas**2)-1)
