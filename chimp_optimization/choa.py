import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy

class ChimpOptimization:
    def __init__(self, n_agents=10, max_iter=15, alpha=0.99):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.alpha = alpha  # Weight for accuracy in fitness
        self.dim = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        # Leaders
        self.attacker_pos = None
        self.attacker_score = -float('inf')
        
        self.barrier_pos = None
        self.barrier_score = -float('inf')
        
        self.chaser_pos = None
        self.chaser_score = -float('inf')
        
        self.driver_pos = None
        self.driver_score = -float('inf')
        
        self.convergence_curve = []

    def fit(self, X, y):
        # Split for fitness evaluation (internal validation)
        # Use a small subset for speed if X is huge, but with 150k limit it's okay
        # Strict constraint: "Use lightweight classifier"
        
        self.dim = X.shape[1]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize positions (continuous values, will be thresholded for binary)
        self.positions = np.random.rand(self.n_agents, self.dim)
        
        # Main Loop
        for t in range(self.max_iter):
            # Update f (decreases linearly from 2.5 to 0)
            f = 2.5 - (t * (2.5 / self.max_iter))
            
            for i in range(self.n_agents):
                # Boundary check (clamp to 0-1 for stability, though not strictly needed for binary logic)
                self.positions[i] = np.clip(self.positions[i], 0, 1)
                
                # Calculate Fitness
                fitness = self._calculate_fitness(self.positions[i])
                
                # Update Leaders
                if fitness > self.attacker_score:
                    self.attacker_score = fitness
                    self.attacker_pos = self.positions[i].copy()
                elif fitness > self.barrier_score:
                    self.barrier_score = fitness
                    self.barrier_pos = self.positions[i].copy()
                elif fitness > self.chaser_score:
                    self.chaser_score = fitness
                    self.chaser_pos = self.positions[i].copy()
                elif fitness > self.driver_score:
                    self.driver_score = fitness
                    self.driver_pos = self.positions[i].copy()
            
            self.convergence_curve.append(self.attacker_score)
            print(f"[ChOA] Iter {t+1}/{self.max_iter} | Best Fitness: {self.attacker_score:.4f}")
            
            # Update positions
            for i in range(self.n_agents):
                # Chimp Groups (randomly assigning strategies or using all 4)
                # Standard ChOA updates position based on all 4 leaders
                
                # Constants for update
                # a = 2 * f * r1 - f
                # c = 2 * r2
                # m = chaotic map val (using simple random here for standard version)
                
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                a = 2 * f * r1 - f
                c = 2 * r2
                
                # D_Attacker = |C * A_pos - m * X| -> assuming m=1 for simplicity or implementing chaotic map
                # Implementing simple chaotic map for 'm' (Chaotic Value)
                # Using Logistic Map: x_new = u * x * (1-x)
                m = np.random.rand(self.dim) # Simplified regular vector
                
                d_attacker = np.abs(c * self.attacker_pos - m * self.positions[i])
                x1 = self.attacker_pos - a * d_attacker
                
                d_barrier = np.abs(c * self.barrier_pos - m * self.positions[i])
                x2 = self.barrier_pos - a * d_barrier
                
                d_chaser = np.abs(c * self.chaser_pos - m * self.positions[i])
                x3 = self.chaser_pos - a * d_chaser
                
                d_driver = np.abs(c * self.driver_pos - m * self.positions[i])
                x4 = self.driver_pos - a * d_driver
                
                # New continuous position
                new_pos = (x1 + x2 + x3 + x4) / 4.0
                
                # Sexual Motivation (Chaotic update)
                # If random prob < 0.5, use chaotic map, else normal update
                if np.random.rand() < 0.5:
                     # Using a simple chaotic scalar perturbation
                     chaotic_val = 0.5 # Placeholder for complex chaotic map
                     new_pos = new_pos + chaotic_val * np.random.randn(self.dim) * 0.1
                
                self.positions[i] = new_pos

        # Return best binary mask
        return self._to_binary(self.attacker_pos), self.convergence_curve

    def _to_binary(self, continuous_pos):
        # Sigmoid transfer function
        sigmoid_vals = 1 / (1 + np.exp(-10 * (continuous_pos - 0.5)))
        # Stochastic binarization
        return (np.random.rand(self.dim) < sigmoid_vals).astype(int)

    def _calculate_fitness(self, continuous_pos):
        # Convert to binary mask
        binary_mask = self._to_binary(continuous_pos)
        
        # If no features selected, return 0 (or select at least 1)
        if np.sum(binary_mask) == 0:
            binary_mask[np.random.randint(0, self.dim)] = 1
            
        selected_indices = np.where(binary_mask == 1)[0]
        
        # Reduced dataset
        X_subset_train = self.X_train[:, selected_indices]
        X_subset_val = self.X_val[:, selected_indices]
        
        # Train Lightweight Classifier (Logistic Regression)
        # Using lbfgs or liblinear, max_iter limited for speed
        clf = LogisticRegression(solver='liblinear', max_iter=100, random_state=42)
        clf.fit(X_subset_train, self.y_train)
        
        acc = clf.score(X_subset_val, self.y_val)
        
        # Fitness Equation: alpha * Acc + (1-alpha) * log10(N/n)
        N = self.dim
        n = len(selected_indices)
        
        # Avoid log(0)
        if n == 0: n = 1
        
        fitness = self.alpha * acc + (1 - self.alpha) * (1 - n/N) # Using simple ratio for stability or log
        # Prompt formula: log10(N/n)
        # If n is small, N/n is large, log is positive. We want to MAXIMIZE fitness?
        # Typically maximize Accuracy and Minimize feature count.
        # If we Add log(N/n), smaller n -> larger log -> larger fitness. Correct.
        
        fitness = self.alpha * acc + (1 - self.alpha) * np.log10(N / n)
        
        return fitness
