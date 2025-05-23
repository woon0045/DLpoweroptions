import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import math

import DataGenerator

import importlib
importlib.reload(DataGenerator)

class DeepNet(nn.ModuleList):
    def __init__(self, sequence_len, hidden_dim, batch_size, fin_model):
        super(DeepNet, self).__init__()

        # initialize financial model
        self.fin_model = fin_model
        # initialize the ML parameters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.dimension = 1

        #Initialize network layers
        # first layer lstm cell
        self.lstm_1 = nn.LSTMCell(input_size=self.dimension, hidden_size=self.hidden_dim)
        # second layer lstm cell
        self.lstm_2 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        # fully connected layer to connect the output of the LSTM cell to the output
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.dimension)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.soft = nn.Softplus(beta=1,threshold=3)

    def forward(self):
        # empty tensor for the control wealth and stock
        output_seq = torch.empty((self.sequence_len, self.batch_size, self.dimension))

        input_seq = torch.empty((self.sequence_len, self.batch_size, self.dimension))

        stock_seq = torch.empty((self.sequence_len, self.batch_size, self.dimension))

        #Generating inputs
        dB = DataGenerator.gen_dB(self.sequence_len,self.batch_size,self.fin_model.dim_BM, self.fin_model.T)

        if self.fin_model.name == "Merton":
            jumps = DataGenerator.gen_jumps_ln(self.sequence_len,self.batch_size,self.fin_model.mus, self.fin_model.sigmas)
            jump_times = DataGenerator.gen_jump_times(self.sequence_len, self.batch_size,
                                                      self.fin_model.rates, self.fin_model.T)
            dN = DataGenerator.gen_compound_poisson(self.sequence_len, self.batch_size, self.fin_model.dim_N,
                                           jumps, jump_times)
            compensator = DataGenerator.gen_compensator_ln(self.fin_model.rates, self.fin_model.mus, self.fin_model.sigmas)

        if self.fin_model.name == "MixedMerton":
            jumps = DataGenerator.gen_jumps_mixture(self.sequence_len, self.batch_size, self.fin_model.mus,
                                               self.fin_model.sigmas, self.fin_model.rates)
            jump_times = DataGenerator.gen_jump_times(self.sequence_len, self.batch_size,
                                                      [sum(self.fin_model.rates)], self.fin_model.T)
            dN = DataGenerator.gen_compound_poisson(self.sequence_len, self.batch_size, self.fin_model.dim_N,
                                                    jumps, jump_times)
            compensator = DataGenerator.gen_compensator_ln(self.fin_model.rates, self.fin_model.mus, self.fin_model.sigmas)

        #Initial stock value
        s = torch.ones(self.batch_size, self.dimension)*self.fin_model.s0

        #Compute initial wealth (option price) with feed-forward NN
        initial_input = torch.ones(self.batch_size, self.hidden_dim)*self.fin_model.s0
        x = self.soft(self.fc(initial_input))

        # init the both layer cells with the zeroth hidden and zeroth cell states
        zeros = torch.zeros(self.batch_size,self.hidden_dim)
        hc_1, hc_2 = (zeros,zeros), (zeros, zeros)

        #Store initial values
        stock_seq[0] = s
        input_seq[0] = x

        #Initialize log(s) and log(x)
        s_log = torch.log(s)
        x_log = torch.log(x)

        # for every timestep use input x[t] to compute control out from hiden state h1 and derive the next imput x[t+1]
        for i in range(self.sequence_len):
            # get the hidden and cell states from the first layer
            hc_1 = self.lstm_1(x, hc_1)
            # unpack the hidden and the cell states from the first layer
            h_1, c_1 = hc_1
            # get the hidden and cell states from the second layer
            hc_2 = self.lstm_2(h_1,hc_2)
            # unpack the hidden and the cell states from the second layer
            h_2, c_2 = hc_2
            #Use hidden state of the second layer as an input of a linear network
            out = self.fc(h_2)
            output_seq[i] = out

            # When working with jump model updating is done using Euler-Maruyama for log(X_t) and log(S_t)
            if i < self.sequence_len - 1:
                if self.fin_model.name != "BlackScholes" and self.fin_model.R < 150:
                    x_log = self.fin_model.new_log_wealth_instance(x_log,out,dB,dN,compensator,i)
                    s_log = self.fin_model.new_log_stock_instance(s_log,dB,dN,compensator,i)
                    x, s = torch.exp(x_log), torch.exp(s_log)  #Obtain stock and wealth values by taking exponential
                elif self.fin_model.name != "BlackScholes" and self.fin_model.R >= 150:
                    x = self.fin_model.new_wealth_instance(x, out, dB, dN, compensator, i)
                    s = self.fin_model.new_stock_instance(s, dB, dN, compensator, i)
                else:
                    x = self.fin_model.new_wealth_instance(x, out, dB, i)
                    s = self.fin_model.new_stock_instance(s, dB, i)

                #Store wealth and stock values
                input_seq[i+1] = x
                stock_seq[i+1] = s

        return output_seq, input_seq, stock_seq, x, s # Tereminal wealth and stock states need to be stored separately
        # for purpose of computing gradients

    def loss(self, x, s):
        F = self.fin_model.claim(s, self.batch_size, self.dimension) #option payoff
        return 0.5 * torch.mean(torch.square(torch.norm(x - F, dim=1))) #aproximation of the minimal variance
        # performance functional


class HedgeSolver():
    def __init__(self, net, learning_rate, epochs):
        self.net = net
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.losses = []
        self.initials = []
        self.time = None
        self.best_state_dict = None

    # Training loop
    def train(self):
        start = time.time()
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate) #Set an Adam optimizer

        best_state_dict = None #Store net's state dictionary at the epoch where minimal loss is obtained
        min_loss = np.inf
        w_neg = 0 #number of negative stock and wealth occurances
        s_neg = 0
        # Training loop
        for epoch in range(self.epochs):
            if epoch % 25 == 0:
                print(f"Epoch {epoch}")

            control, wealth, stock, X_T, S_T = self.net()

            loss = self.net.loss(X_T, S_T)
            if math.isnan(loss):
                return control, wealth, stock, w_neg, s_neg

            #If negative wealth or stock values occur skip this epoch
            if torch.min(wealth) <=0 or torch.min(stock) <= 0:
                w_neg += (torch.min(wealth) <=0)
                s_neg += (torch.min(stock) <= 0)
                continue

            loss.backward()
            optimizer.step()
            self.net.zero_grad()

            self.losses = np.append(self.losses, loss.detach().cpu().numpy())
            self.initials = np.append(self.initials, wealth[0,0,0].detach().cpu().numpy())
            if (epoch > 0.8*self.epochs) and (loss < min_loss):
                min_loss = loss
                best_state_dict = self.net.state_dict()
        end = time.time()
        self.time = (end-start)//60
        self.best_state_dict = best_state_dict
        return w_neg, s_neg

class HedgeEvaluator():
    def __init__(self, net, fin_model):
        self.net = net
        self.fin_model = fin_model
        self.control = None
        self.wealth = None
        self.stock = None
        self.price = None
        self.claim = None
        self.loss = None
        self.l2 = None

    def eval(self):
        self.control, self.wealth, self.stock, X_T, S_T = self.net()
        self.price = float(self.wealth[0,0,0])
        self.loss = float(self.net.loss(X_T, S_T))
        self.claim = self.fin_model.claim(S_T, self.net.batch_size, self.net.dimension)

        if self.fin_model.name == "BlackScholes":
            portfolio = self.fin_model.black_scholes_portfolio(self.stock, self.wealth)
            self.l2 = torch.mean(torch.sqrt(torch.sum((self.control[:,:,0] - portfolio[:,:,0])**2, dim=0)), dim=0)
