import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import Counter
import os
from numba import jit

class AdaptiveMIRTNew:
    def __init__(self, select_noise=0.1, n_items=1000, n_traits=6, n_steps=5, probs=None, verbose=False):
        if probs is None:
            probs = [0.4, 0.2, 0.2, 0.1, 0.1]
        self.verbose = verbose
        item_opts = ['likert', 'binary', 'value', 'mc_single', 'mc_multi']
        self.item_types = np.random.choice(item_opts, size=n_items, p=probs)
        
        self.n_items = n_items
        self.n_traits = n_traits
        self.n_steps = n_steps
        
        self.true_th = np.random.uniform(-3, 3, size=n_traits)
        self.est_th = np.zeros(n_traits)
        self.th_hist = []
        
        # Pre-compute parameters as numpy arrays
        self.a_params = np.random.randn(n_items, n_traits)
        self.thresholds = np.sort(np.random.uniform(-2, 2, size=(n_items, 4)), axis=1)
        self.bin_b = np.random.randn(n_items)
        self.val_thresh = np.sort(np.random.uniform(-2, 2, size=5))
        self.mc_params = np.random.randn(n_items, n_traits, 4)
        
        self.sel_items = []
        self.responses = []
        self.info_gain = []
        self.bounds = [(-3, 3)] * n_traits
        self.last_item = None
        self.select_noise = select_noise

    @staticmethod
    @jit(nopython=True)
    def binary_prob(a, b, th):
        """Standard binary probability"""
        return 1 / (1 + np.exp(-(np.dot(a, th) - b)))

    @staticmethod
    @jit(nopython=True)
    def mc_binary_prob(a, th):
        """MC multi binary probability"""
        logits = np.dot(a.T, th)
        return 1 / (1 + np.exp(-logits))

    @staticmethod
    @jit(nopython=True)
    def gpcm_prob(a, th, thresholds):
        diff = np.dot(a, th)
        probs = np.ones(len(thresholds) + 1)
        probs[1:] = np.exp(diff - thresholds)
        return probs / np.sum(probs)

    @staticmethod
    @jit(nopython=True)
    def mc_single_prob(a, th):
        expnt = np.dot(a.T, th)
        max_expnt = np.max(expnt)
        exp_expnt = np.exp(expnt - max_expnt)
        return exp_expnt / np.sum(exp_expnt)

    def _item_log_like(self, th, item, resp=None):
        it_type = self.item_types[item]
        
        if it_type == "binary":
            prob = self.binary_prob(self.a_params[item], self.bin_b[item], th)
            if resp is None:
                return prob * (1 - prob)
            prob = np.clip(prob, 1e-8, 1 - 1e-8)
            return resp * np.log(prob) + (1 - resp) * np.log(1 - prob)
            
        elif it_type == "likert":
            probs = self.gpcm_prob(self.a_params[item], th, self.thresholds[item])
            if resp is None:
                return np.sum(probs * (1 - probs))
            return np.log(probs[resp - 1])
            
        elif it_type == "value":
            probs = self.gpcm_prob(self.a_params[item], th, self.val_thresh)
            if resp is None:
                return np.sum(probs * (1 - probs))
            return np.log(probs[resp])
            
        elif it_type == "mc_single":
            probs = self.mc_single_prob(self.mc_params[item], th)
            if resp is None:
                return np.sum(probs * (1 - probs))
            return np.log(probs[resp])
            
        else:  # mc_multi
            probs = self.mc_binary_prob(self.mc_params[item], th)  # Using separate function for mc_multi
            probs = np.clip(probs, 1e-8, 1 - 1e-8)
            if resp is None:
                return np.sum(probs * (1 - probs))
            return np.sum([resp[j] * np.log(probs[j]) + (1 - resp[j]) * np.log(1 - probs[j]) 
                         for j in range(len(resp))])

    def log_like(self, th, item=None):
        if item is not None:
            return self._item_log_like(th, item)
        
        ll = 0
        for i, q in enumerate(self.sel_items):
            ll += self._item_log_like(th, q, self.responses[i])
        return -ll

    def next_item(self):
        infos = np.full(self.n_items, -np.inf)
        remaining = np.setdiff1d(np.arange(self.n_items), self.sel_items)
        infos[remaining] = [self.log_like(self.est_th, item=i) for i in remaining]
        
        if self.select_noise > 0:
            infos[remaining] += np.random.randn(len(remaining)) * self.select_noise
            
        next_item = np.argmax(infos)
        self.sel_items.append(next_item)
        self.last_item = next_item
        
        if self.verbose:
            print(f"Selected Item {next_item+1}")
        return next_item

    def sim_resp(self):
        if self.last_item is None:
            raise ValueError("No item selected.")
            
        q = self.last_item
        it_type = self.item_types[q]
        
        if it_type == "binary":
            prob = self.binary_prob(self.a_params[q], self.bin_b[q], self.true_th)
            resp = np.random.binomial(1, prob)
            
        elif it_type == "likert":
            probs = self.gpcm_prob(self.a_params[q], self.true_th, self.thresholds[q])
            resp = np.argmax(np.random.multinomial(1, probs)) + 1
            
        elif it_type == "value":
            probs = self.gpcm_prob(self.a_params[q], self.true_th, self.val_thresh)
            resp = np.argmax(np.random.multinomial(1, probs))
            
        elif it_type == "mc_single":
            probs = self.mc_single_prob(self.mc_params[q], self.true_th)
            resp = np.argmax(np.random.multinomial(1, probs))
            
        else:  # mc_multi
            probs = self.mc_binary_prob(self.mc_params[q], self.true_th)
            resp = np.random.binomial(1, probs)
        
        self.responses.append(resp)
        self.info_gain.append(self.log_like(self.est_th, item=q))
        
        if self.verbose:
            print(f"Simulated Response: {resp}, Info Gain: {self.info_gain[-1]}")
        return resp

    def update_theta(self):
        res = minimize(self.log_like, self.est_th, method='L-BFGS-B', bounds=self.bounds)
        self.est_th = res.x[:self.n_traits]
        self.th_hist.append(self.est_th.copy())
        
        if self.verbose:
            print(f"Updated Theta: {self.est_th}")

    def _get_theta(self):
        if len(self.est_th):
            return self.est_th
        else:
            return self.true_th

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
            n_rows = (self.n_traits + 1) // 2
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
            fig1.savefig(os.path.join(save_dir, f"MIRT_info_{self.select_noise:.2f}.png"))
            fig2.savefig(os.path.join(save_dir, f"MIRT_theta_{self.select_noise:.2f}.png"))