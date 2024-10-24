import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class AdaptiveMIRT:
    def __init__(self, n_items=1000, n_traits=6, n_steps=5, probs=None, verbose=False):
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
        self.a_params = np.random.randn(n_items, n_traits)
        self.thresholds = [np.sort(np.random.uniform(-2, 2, size=4)) for _ in range(n_items)]
        self.bin_b = np.random.randn(n_items)
        self.val_thresh = np.sort(np.random.uniform(-2, 2, size=5))
        self.mc_params = np.random.randn(n_items, n_traits, 4)
        self.sel_items = []
        self.responses = []
        self.info_gain = []
        self.bounds = [(-3, 3)] * n_traits
        self.last_item = None

    def log_lik(self, th, item=None):
        if item is not None:
            return self._item_log_lik(th, item)
        else:
            ll = 0
            for i, q in enumerate(self.sel_items):
                ll += self._item_log_lik(th, q, self.responses[i])
            return -ll

    def _item_log_lik(self, th, item, resp=None):
        it_type = self.item_types[item]
        if it_type == "binary":
            prob = self.bin_prob(self.a_params[item], self.bin_b[item], th)
            if resp is None:
                return prob * (1 - prob)
            else:
                prob = np.clip(prob, 1e-8, 1 - 1e-8)
                return resp * np.log(prob) + (1 - resp) * np.log(1 - prob)
        elif it_type == "likert":
            probs = self.gpcm_prob(self.a_params[item], th, self.thresholds[item])
            return np.sum(probs * (1 - probs)) if resp is None else np.log(probs[resp - 1])
        elif it_type == "value":
            probs = self.gpcm_prob(self.a_params[item], th, self.val_thresh)
            return np.sum(probs * (1 - probs)) if resp is None else np.log(probs[resp])
        elif it_type == "mc_single":
            probs = self.mc_single_prob(self.mc_params[item], th)
            return np.sum(probs * (1 - probs)) if resp is None else np.log(probs[resp])
        elif it_type == "mc_multi":
            probs = self.mc_multi_prob(self.mc_params[item], th)
            return np.sum(probs * (1 - probs)) if resp is None else sum([resp[j] * np.log(probs[j]) + (1 - resp[j]) * np.log(1 - probs[j]) for j in range(len(resp))])

    def next_item(self):
        infos = []
        for i in range(self.n_items):
            if i in self.sel_items:
                infos.append(-np.inf)
                continue
            info = self.log_lik(self.est_th, item=i)
            infos.append(info + np.random.randn() * 0.1)
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
            prob = self.bin_prob(self.a_params[q], self.bin_b[q], self.true_th)
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
        elif it_type == "mc_multi":
            probs = self.mc_multi_prob(self.mc_params[q], self.true_th)
            resp = np.random.binomial(1, probs)
        self.responses.append(resp)
        self.info_gain.append(self.log_lik(self.est_th, item=q))
        if self.verbose:
            print(f"Simulated Response: {resp}, Info Gain: {self.info_gain[-1]}")
        return resp

    def bin_prob(self, a, b, th):
        return expit(np.dot(a, th) - b)

    def gpcm_prob(self, a, th, thresholds):
        diff = np.dot(a, th)
        probs = [1] + [np.exp(diff - thresholds[k]) for k in range(len(thresholds))]
        return np.array(probs) / np.sum(probs)

    def mc_single_prob(self, a, th):
        expnt = np.dot(a.T, th)
        return np.exp(expnt) / np.sum(np.exp(expnt))

    def mc_multi_prob(self, a, th):
        probs = expit(np.dot(a.T, th))
        return np.clip(probs, 0, 1)

    def plot_results(self):
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))
        axs[0].plot(self.info_gain, marker='o', linestyle='-', color='b')
        axs[0].set_title("Info Gain During Adaptive Testing")
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("Info Gain")
        axs[0].grid(True)
        for i in range(self.n_traits):
            axs[1].scatter([i+1], [self.true_th[i]], color='green', label='True' if i == 0 else "")
            axs[1].scatter([i+1], [self.est_th[i]], color='red', label='Est' if i == 0 else "")
        axs[1].set_title("Final Est. vs True Traits")
        axs[1].set_xlabel("Trait")
        axs[1].set_ylabel("Value")
        axs[1].legend()
        axs[1].grid(True)
        from collections import Counter
        sel_it_types = [self.item_types[q] for q in self.sel_items]
        it_counts = Counter(sel_it_types)
        axs[2].bar(it_counts.keys(), it_counts.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
        axs[2].set_title("Item Types Selected")
        axs[2].set_xlabel("Type")
        axs[2].set_ylabel("Count")
        axs[2].grid(True)
        th_hist = np.array(self.th_hist)
        for i in range(self.n_traits):
            axs[3].plot(th_hist[:, i], label=f"Est. Trait {i+1}")
            axs[3].axhline(self.true_th[i], color='green', linestyle='--', label=f"True Trait {i+1}" if i == 0 else "")
        axs[3].set_title("Theta Est. Change vs True Theta")
        axs[3].set_xlabel("Step")
        axs[3].set_ylabel("Theta Value")
        axs[3].legend()
        axs[3].grid(True)
        plt.tight_layout()
        plt.show()

    def update_theta(self):
        res = minimize(self.log_lik, self.est_th, method='L-BFGS-B', bounds=self.bounds)
        self.est_th = res.x[:self.n_traits]
        self.th_hist.append(self.est_th.copy())
        print(f"Updated Theta: {self.est_th}")
