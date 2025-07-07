import numpy as np
from .FedAvg import FedAvg

class SCAFFOLD(FedAvg):
    def __init__(self, lr_server=1.0):
        super().__init__()
        self.name = "SCAFFOLD"
        self.lr_server = lr_server
        self.global_control = None
        self.client_controls = {}
        self.round_num = 0
        print(f"SCAFFOLD initialized with server_lr={lr_server}")
    
    def aggregate(self, model_weights_list, sample_sizes=None, client_updates=None):
        self.round_num += 1
        
        if client_updates is None:
            # Fallback to standard FedAvg if no SCAFFOLD updates
            return super().aggregate(model_weights_list, sample_sizes)
        
        # Extract weights and control updates from client data
        weights_list = []
        control_updates = []
        participating_clients = []
        
        for i, update_data in enumerate(client_updates):
            if isinstance(update_data, dict):
                weights_list.append(update_data['weights'])
                control_updates.append(update_data['control_update'])
                participating_clients.append(update_data.get('client_id', i))
            else:
                weights_list.append(update_data)
                control_updates.append(None)
                participating_clients.append(i)
        
        # Standard FedAvg aggregation for weights
        aggregated_weights = super().aggregate(weights_list, sample_sizes)
        
        # Initialize global control if first round
        if self.global_control is None:
            self.global_control = [np.zeros_like(w) for w in aggregated_weights]
        
        # Update global control variate
        if any(cu is not None for cu in control_updates):
            if sample_sizes is None:
                sample_sizes = [1] * len(control_updates)
            
            total_samples = sum(sample_sizes)
            
            # Aggregate control updates
            aggregated_control_update = [np.zeros_like(gc) for gc in self.global_control]
            
            for i, (control_update, n_samples) in enumerate(zip(control_updates, sample_sizes)):
                if control_update is not None:
                    weight = n_samples / total_samples
                    for j, cu in enumerate(control_update):
                        aggregated_control_update[j] += weight * cu
            
            # Update global control: c^{t+1} = c^t + \eta_s * \Delta c
            for i in range(len(self.global_control)):
                self.global_control[i] += self.lr_server * aggregated_control_update[i]
        
        print(f"SCAFFOLD Round {self.round_num}: Aggregated {len(weights_list)} clients")
        return aggregated_weights
    
    def get_global_control(self):
        return self.global_control
    
    def get_client_control(self, client_id):
        if client_id not in self.client_controls:
            if self.global_control is not None:
                self.client_controls[client_id] = [np.zeros_like(gc) for gc in self.global_control]
            else:
                self.client_controls[client_id] = None
        return self.client_controls[client_id]
    
    def update_client_control(self, client_id, new_control):
        self.client_controls[client_id] = new_control
