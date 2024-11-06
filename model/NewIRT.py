import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import Counter
import os

class AdaptiveMIRTv2:
    def __init__(self, select_noise=0.1, n_items=1000, n_traits=6, probs=None, verbose=False):
        if probs is None:
            probs = [0.4, 0.2, 0.2, 0.1, 0.1]
        self.verbose = verbose
        self.item_opts = ['likert', 'binary', 'value', 'mc_single', 'mc_multi']
        
        # Assign types to items
        self.item_types = np.random.choice(self.item_opts, size=n_items, p=probs)
        
        # Thresholds for Likert-type items (one set per item)
        self.thresholds = np.sort(np.random.uniform(-2, 2, size=(n_items, 4)), axis=1)
        
        # Thresholds for Value-type items (also one set per item for flexibility)
        self.val_thresh = np.sort(np.random.uniform(-2, 2, size=(n_items, 5)), axis=1)
        
        # Initialize parameters
        self.n_items = n_items
        self.n_traits = n_traits
        self.true_th = np.random.uniform(-3, 3, size=n_traits)
        self.est_th = np.zeros(n_traits)
        self.th_hist = []
        self.a_params = np.random.randn(n_items, n_traits)
        self.bin_b = np.random.randn(n_items)
        self.mc_params = np.random.randn(n_items, n_traits, 4)
        
        # Lists for storing selected items, responses, and information gains
        self.sel_items = []
        self.responses = []
        self.info_gain = []
        
        # Hyperparameters
        self.bounds = [(-3, 3)] * n_traits
        self.last_item = None
        self.select_noise = select_noise

    # Probability function for binary items
    def bin_prob(self, a, b, theta):
        return expit(a @ theta - b)

    # Likert scale probability function using GPCM
    def scale_prob(self, a, theta, thresholds):
        diff = a @ theta
        probs = np.exp(np.array([diff - t for t in thresholds]))
        return probs / np.sum(probs)

    # Wrapper for MC Single Answer Probability (Nominal Response Model)
    def mc_single_prob(self, mc_params, theta):
        logits = mc_params.T @ theta
        return expit(logits)

    # Wrapper for MC Multi-Answer Probability (Binary Logistic Model for each option)
    def mc_multi_prob(self, mc_params, theta):
        logits = mc_params.T @ theta
        return expit(logits)  # Applies sigmoid individually for each option

    # Vectorized Log-Likelihood Calculation
    def log_like(self, theta):
        ll_binary = np.sum([
            self.responses[i] * np.log(self.bin_prob(self.a_params[q], self.bin_b[q], theta) + 1e-8) +
            (1 - self.responses[i]) * np.log(1 - self.bin_prob(self.a_params[q], self.bin_b[q], theta) + 1e-8)
            for i, q in enumerate(self.binary_items)
        ])
        # Additional log-likelihoods for other item types would be added similarly
        return -ll_binary

    # Optimized item selection with vectorized noise and information calculation
    def next_item(self):
        infos = np.array([self.log_like(self.est_th) + np.random.randn() * self.select_noise for _ in range(self.n_items)])
        next_item = np.argmax(infos)
        self.sel_items.append(next_item)  # Append to list
        self.last_item = next_item
        return next_item

    def update_theta(self):
        res = minimize(self.log_like, self.est_th, method='L-BFGS-B', bounds=self.bounds)
        self.est_th = res.x[:self.n_traits]
        self.th_hist.append(self.est_th.copy())
        if self.verbose:
            print(f"Updated Theta: {self.est_th}")

    # Simulation of response for the selected item
    def simulate_response(self):
        if self.last_item is None:
            raise ValueError("No item selected.")
        q = self.last_item
        it_type = self.item_types[q]
        if it_type == "binary":
            prob = self.bin_prob(self.a_params[q], self.bin_b[q], self.true_th)
            resp = np.random.binomial(1, prob)
        elif it_type == "likert":
            probs = self.scale_prob(self.a_params[q], self.true_th, self.thresholds[q])
            resp = np.argmax(np.random.multinomial(1, probs)) + 1
        elif it_type == "value":
            probs = self.scale_prob(self.a_params[q], self.true_th, self.val_thresh[q])
            resp = np.argmax(np.random.multinomial(1, probs))
        elif it_type == "mc_single":
            probs = self.mc_single_prob(self.mc_params[q], self.true_th)
            resp = np.argmax(np.random.multinomial(1, probs))
        elif it_type == "mc_multi":
            probs = self.mc_multi_prob(self.mc_params[q], self.true_th)
            resp = np.random.binomial(1, probs)
        self.responses.append(resp)
        self.info_gain.append(self.log_like(self.est_th))
        return resp
    
    
    def plot_results(self, plot_info=True, plot_theta=True, no_show=False, save_fig=True, save_dir="figure"):
        if plot_info:
            fig1, axs1 = plt.subplots(3, 1, figsize=(12, 15))
            axs1[0].plot(self.info_gain, marker='o', linestyle='-', color='b')
            axs1[0].set_title(f"Information Gain During Adaptive Testing (noise={self.select_noise})")
            axs1[0].set_xlabel("Step")
            axs1[0].set_ylabel("Info Gain")
            axs1[0].grid(True)

            for i in range(self.n_traits):
                axs1[1].scatter([i+1], [self.true_th[i]], color='green', label='True' if i == 0 else "")
                axs1[1].scatter([i+1], [self.est_th[i]], color='red', label='Est' if i == 0 else "")
            axs1[1].set_title(f"Final Latent Trait Estimates vs. True Traits (noise={self.select_noise})")
            axs1[1].set_xlabel("Trait Number")
            axs1[1].set_ylabel("Trait Value")
            axs1[1].legend()
            axs1[1].grid(True)

            sel_it_types = [self.item_types[q] for q in self.sel_items]
            it_counts = Counter(sel_it_types)
            axs1[2].bar(it_counts.keys(), it_counts.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
            axs1[2].set_title(f"Distribution of Item Types Selected During Adaptive Testing (noise={self.select_noise})")            
            axs1[2].set_xlabel("Item Type")
            axs1[2].set_ylabel("Count of Selected Items")
            axs1[2].grid(True)
            fig1.tight_layout()
            if not no_show:
                plt.show()
            plt.close(fig1)

        
        if plot_theta:
            n_rows = (self.n_traits + 1) // 2  # Number of rows needed for 3x2 layout
            fig2, axs2 = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
            th_hist = np.array(self.th_hist)
            for i in range(self.n_traits):
                row, col = divmod(i, 2)
                axs2[row, col].plot(th_hist[:, i], label=f"Estimated Trait {i+1}")
                axs2[row, col].axhline(self.true_th[i], color='green', linestyle='--', label=f"True Trait {i+1}")
                axs2[row, col].set_title(f"Theta Estimation Change for Trait {i+1} (noise={self.select_noise})")
                axs2[row, col].set_xlabel("Step")
                axs2[row, col].set_ylabel("Theta Value")
                axs2[row, col].legend()
                axs2[row, col].grid(True)
            

            

            if self.n_traits % 2 != 0:
                fig2.delaxes(axs2[-1, -1])
            fig2.tight_layout()
            if not no_show:
                plt.show()
            plt.close(fig2)


        if save_fig:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            fig1.savefig(os.path.join(save_dir, 
                                      f"MIRT_info_{self.select_noise:.2f}.png"))
            fig2.savefig(os.path.join(save_dir, 
                                      f"MIRT_theta_{self.select_noise:.2f}.png"))
