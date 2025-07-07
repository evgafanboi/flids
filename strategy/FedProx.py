import numpy as np
from .FedAvg import FedAvg

class FedProx(FedAvg):
    def __init__(self, mu=0.01, adaptive_mu=True, warmup_rounds=3, delayed_start=True):
        super().__init__()
        self.name = "FedProx"
        self.base_mu = mu
        self.adaptive_mu = adaptive_mu
        self.warmup_rounds = warmup_rounds
        self.delayed_start = delayed_start
        self.current_round = 0
        self.global_weights = None
        print(f"FedProx initialized with base_mu={mu}, adaptive={adaptive_mu}, warmup_rounds={warmup_rounds}, delayed_start={delayed_start}")
    
    @property
    def mu(self):
        if self.delayed_start and self.current_round <= 2:
            return 0.0  # No proximal term for first 2 rounds
        
        if not self.adaptive_mu:
            return self.base_mu
        
        effective_round = max(0, self.current_round - (2 if self.delayed_start else 0))
        
        if effective_round <= self.warmup_rounds:
            # Linear warmup: start at 0, reach base_mu at warmup_rounds
            return self.base_mu * (effective_round / self.warmup_rounds) if self.warmup_rounds > 0 else self.base_mu
        else:
            # After warmup, gradually increase
            growth_factor = 1 + 0.1 * (effective_round - self.warmup_rounds)
            return min(self.base_mu * growth_factor, self.base_mu * 3.0)  # Cap at 3x base
    
    def aggregate(self, model_weights_list, sample_sizes=None):
        self.current_round += 1
        current_mu = self.mu
        print(f"FedProx Round {self.current_round}: Using μ = {current_mu:.4f}")
        
        aggregated_weights = super().aggregate(model_weights_list, sample_sizes)
        self.global_weights = [w.copy() for w in aggregated_weights]
        return aggregated_weights
