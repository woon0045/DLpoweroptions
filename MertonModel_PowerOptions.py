#######################################
#Here you can set desired financial and machine learning parameters in dictionaries
#######################################
# Financial model parameters

financial_parameters = {
    'T': 1.0,                  # Maturity time
    'drift': [0.2],            # Drift
    'volatility': [0.2],       # Volatility
    's0': 1.0,                 # Spot price
    'r': 0,                    # Interest rate
    'K': 0.5,                  # Strike price
    'p': 1.1,                  # Power option exponent
    'rates': [3.0, 5.0, 2.0],            # Rates for Poisson process
    'mus': [-0.1, -0.1, -0.05],          # Mean log-normal parameters
    'sigmas': [0.05, 0.02, 0.01],        # Standard deviation log-normal parameters
    'MC_sample_size': 5000,    # Sample size for MC
    'MC_R': 3000               # Number of time steps for MC
}
if len(financial_parameters["rates"])!=len(financial_parameters["mus"]) or \
        len(financial_parameters["rates"])!=len(financial_parameters["sigmas"]):
    raise Exception("Dimension of rate and jump parameters must coincide")

#######################################
# Machine learning model parameters

ml_parameters = {
    'M': 256,                    # Batch size
    'hidden_dim': 512,           # Number of neurons at each LSTM layer
    'R': 160,                    # Number of time-steps
    'epochs': 8000,              # Number of epochs (training iterations)
    'learning_rate': 0.0005,     # Learning rate
    'eval_size': 10000           # Size of the evaluation set
}

#######################################

#Once parameters are set run this file


#######################################
#Creating paths and folders
#######################################

market_name = "Merton"

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M")

#formatted_datetime = "2023_11_30_17_50"

market_path = market_name+"/"+ formatted_datetime +"/"

if not os.path.exists(market_path):
    os.makedirs(market_path)

graph_path = market_path + "Graphs/"
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

model_path = market_path  + "Model/"
if not os.path.exists(model_path):
    os.makedirs(model_path)


########################################
# Merton class
#######################################
class Merton():
    def __init__(self, market_name, T, drift, volatility, s0, r, rates, mus, sigmas, K, p, R, MC_sample_size, MC_R):
        self.name = market_name
        self.T = T
        self.drift = torch.tensor(drift).unsqueeze(-1).float()
        self.volatility = torch.tensor(volatility).unsqueeze(-1).float()
        self.dim_BM = len(volatility)
        self.s0 = s0
        self.r = r
        self.rates = rates
        self.mus = mus
        self.sigmas = sigmas
        self.dim_N = len(rates)
        self.K = K
        self.p = p
        self.R = R
        self.dt = T / R
        self.initial = self.initial_value_MC(MC_sample_size, MC_R)

    # Computes new instance of wealth according to the update rule prescribed by Euler-Maruyama scheme
    def new_wealth_instance(self, x, control, dB, dN, compensator, i):
        drift_comp = self.drift - torch.sum(compensator, dim=0, keepdim=True)  # compensated drift
        dx = x * ((1 - control) * self.r + control * drift_comp) * self.dt + \
             x * control * torch.matmul(dB[i, :, :], self.volatility) + \
             x * control * torch.sum(dN[i, :, :], dim=-1, keepdim=True)
        return x + dx

    # Computes new instance of wealth according to the update rule prescribed by Euler-Maruyama scheme
    def new_stock_instance(self, s, dB, dN, compensator, i):
        drift_comp = self.drift - torch.sum(compensator, dim=0, keepdim=True)  # compensated drift
        ds = s * drift_comp * self.dt + \
             s * torch.matmul(dB[i, :, :], self.volatility) + \
             s * torch.sum(dN[i, :, :], dim=-1, keepdim=True)
        return s + ds

    # Computes new instance of wealth logarithm according to the update rule prescribed by Euler-Maruyama scheme
    def new_log_wealth_instance(self, x, control, dB, dN, compensator, i):
        drift_comp = self.drift - torch.sum(compensator, dim=0, keepdim=True)  # compensated drift
        volatility_t = torch.transpose(self.volatility, 0, 1)
        tmp = control * dN[i, :, :]
        tmp[tmp < -1] = -0.99
        dx = ((1 - control) * self.r + control * drift_comp - 0.5 * control ** 2 *
              torch.inner(volatility_t, volatility_t)) * self.dt + \
             control * torch.matmul(dB[i, :, :], self.volatility) + \
             torch.sum(torch.log(tmp + 1), dim=-1, keepdim=True)
        return x + dx

    # Computes new instance of stock logarithm according to the update rule prescribed by Euler-Maruyama scheme
    def new_log_stock_instance(self, s, dB, dN, compensator, i):
        drift_comp = self.drift - torch.sum(compensator, dim=0, keepdim=True)  # compensated drift
        volatility_t = torch.transpose(self.volatility, 0, 1)
        ds = (drift_comp - 0.5 * torch.inner(volatility_t, volatility_t)) * self.dt + \
             torch.matmul(dB[i, :, :], self.volatility) + \
             torch.sum(torch.log(dN[i, :, :] + 1), dim=-1, keepdim=True)
        return s + ds

    # Computes the option claims given terminal stock price and dimensions of assets. This is for the case of
    # Power option
    def claim(self, s, dim1, dim2):
        K_tensor = torch.ones(dim1, dim2) * self.K
        return torch.max(torch.zeros(dim1, dim2), s ** self.p - K_tensor)

    # Monte-Carlo estimation of the initial wealth
    def initial_value_MC(self, sample_size, R):
        # Generate stock realizations
        B_T = DataGenerator.gen_B_t(sample_size, self.dim_BM, self.T)
        jumps = DataGenerator.gen_jumps_ln(R, sample_size, self.mus, self.sigmas)
        jump_times = DataGenerator.gen_jump_times(R, sample_size,
                                                  self.rates, self.T)
        dN = DataGenerator.gen_compound_poisson(R, sample_size, self.dim_N,
                                                jumps, jump_times)
        compensator = DataGenerator.gen_compensator_ln(self.rates, self.mus, self.sigmas)

        drift_comp = self.drift - torch.sum(compensator, dim=0, keepdim=True)  # compensated drift
        volatility_t = torch.transpose(self.volatility, 0, 1)
        s_T = self.s0 * torch.exp((drift_comp - 0.5 * torch.inner(volatility_t, volatility_t)) * self.T + \
                                  torch.matmul(B_T, self.volatility) + \
                                  torch.sum(torch.sum(torch.log(dN + 1), dim=-1, keepdim=True), dim=0))

        # Power option claim
        F = self.claim(s_T, sample_size, 1)

        # Generate Radon-Nykodin derivative Z realizations
        G = self.G()
        dN_Z = DataGenerator.gen_compound_poisson(R, sample_size, self.dim_N, G*jumps, jump_times)

        Z_T = torch.exp((-0.5 * (G ** 2) * torch.inner(volatility_t, volatility_t) -
                         G * torch.sum(compensator, dim=0, keepdim=True)) * self.T + \
                        G * torch.matmul(B_T, self.volatility) + \
                        torch.sum(torch.sum(torch.log(1 + dN_Z), dim=-1, keepdim=True), dim=0))

        dN_Z2 = DataGenerator.gen_compound_poisson(R, sample_size, self.dim_N, torch.log(1+G * jumps), jump_times)

        Z_T2 = torch.exp((-0.5 * (G ** 2) * torch.inner(volatility_t, volatility_t) -
                         G * torch.sum(compensator, dim=0, keepdim=True)) * self.T + \
                        G * torch.matmul(B_T, self.volatility) + \
                        torch.sum(torch.sum(torch.log(dN_Z2), dim=-1, keepdim=True), dim=0))

        return torch.mean(F * Z_T)

    def G(self):
        volatility_t = torch.transpose(self.volatility, 0, 1)
        mus_t = torch.tensor(self.mus).float()
        sigmas_t = torch.tensor(self.sigmas).float()
        rates_t = torch.tensor(self.rates).float()
        ln_m1 = torch.exp(mus_t + 0.5 * sigmas_t ** 2) - 1  # first moment of log-normal distribution -1
        ln_m2 = torch.exp(2 * mus_t + sigmas_t ** 2) * (torch.exp(sigmas_t ** 2) - 1) + ln_m1 ** 2  # second
        # moment of log-normal distribution
        G = - self.drift / (torch.inner(volatility_t, volatility_t) + torch.inner(ln_m2, rates_t))
        return G


#################################################
# Training
#################################################

# Initialize financial model
fin_model = Merton("Merton", financial_parameters["T"], financial_parameters["drift"],
                   financial_parameters["volatility"], financial_parameters["s0"], financial_parameters["r"],
                   financial_parameters["rates"], financial_parameters["mus"], financial_parameters["sigmas"],
                   financial_parameters["K"], financial_parameters["p"], ml_parameters["R"], financial_parameters["MC_sample_size"],
                   financial_parameters["MC_R"])

#Initialize deep neural network
net = DeepSolver.DeepNet(ml_parameters["R"],ml_parameters["hidden_dim"],ml_parameters["M"],fin_model)

#Initialize the solver and train the model
solver = DeepSolver.HedgeSolver(net,ml_parameters["learning_rate"],ml_parameters["epochs"])
errs = solver.train()

#Save model weights
torch.save(solver.net.state_dict(), model_path + "weights")
np.save(model_path+"losses", solver.losses)
np.save(model_path + "initials", solver.initials)

###################################################
#Evaluation
###################################################
with torch.no_grad():
    #Initialize evaluation net
    net_eval = DeepSolver.DeepNet(ml_parameters["R"], ml_parameters["hidden_dim"], ml_parameters["eval_size"], fin_model)

    #Load model weights and set in evaluation mode
    net_eval.load_state_dict(torch.load(model_path + "weights"))
    net_eval.eval()

    evaluator = DeepSolver.HedgeEvaluator(net_eval, fin_model)
    evaluator.eval()


# Make a text file with financial an ML information, training time and evaluated loss

with open(market_path + 'Info.txt', 'w') as file:
    # Write financial_data to the file
    file.write("Financial parameters:\n")
    for key, value in financial_parameters.items():
        file.write(f"{key}: {value}\n")

    # Add a separator between the two dictionaries
    file.write("\n---\n\n")

    # Write lstm_data to the file
    file.write("Machine learning parameters:\n")
    for key, value in ml_parameters.items():
        file.write(f"{key}: {value}\n")

    file.write("\n---\n\n")
    file.write("Loss: " + str(evaluator.loss))

    file.write("\n---\n\n")
    file.write("Initial: " + str(float(evaluator.wealth[0, 0, 0])))

    file.write("\n---\n\n")
    file.write("Theoretical Initial: " + str(float(fin_model.initial)))

    file.write("\n---\n\n")
    file.write("Training time: " + str(solver.time) + " min")


with open(market_path + 'Info.txt', 'r') as file:
    content = file.read()

print(content)

#######################################################################################################
#Make and save plots

ploter = Plots.Plots(fin_model, solver, evaluator, graph_path)

save = True #Whether you want to save the graphs
start = 30

ploter.plot_losses(start=start, save=save)
ploter.plot_initials(start=start,save=save)

i = np.random.randint(ml_parameters["eval_size"])
ploter.plot_market(i, save=save)
