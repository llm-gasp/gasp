import torch
import numpy as np
from sklearn.manifold import TSNE
from skopt import gp_minimize

class Embeddings:
    def __init__(self, model, tokenizer, reduced_dim, seed):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.reduced_dim = reduced_dim
        self.seed = seed
    
    def get_embeddings(self, suffixes):
        embeddings = []
        with torch.no_grad():
            for suffix in suffixes:
                inputs = self.tokenizer(suffix, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
                outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                suffix_embedding = hidden_states.mean(dim=1).squeeze().detach().cpu().numpy()
                embeddings.append(suffix_embedding)
        return np.array(embeddings)
    
    def dimensionality_reduction(self, embeddings):
        tsne = TSNE(n_components=self.reduced_dim, perplexity=min(max(embeddings.shape[0] - 1, 1), 30), random_state=self.seed)  
        reduced_embeddings = tsne.fit_transform(embeddings)

        return reduced_embeddings
    
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))

    def find_closest_neighbor(self, target_point, points):
        min_distance = float('inf')
        closest_point = None
        for point in points:
            distance = self.euclidean_distance(target_point, point)
            if distance < min_distance:
                min_distance = distance
                closest_point = point
        return closest_point, min_distance
    
class LBO:
    def __init__(self, config, model, tokenizer, blackbox, evaluator):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.blackbox = blackbox
        self.evaluator = evaluator
        self.searched_points = {}
        self.responses = {}
        self.PROMPT_LBO = ""
        self.MAPPING_LBO = {}
        
        self.acq_fn = self.config["lbo_params"]["acquisition_function"]
        self.n_calls = self.config["lbo_params"]["n_calls"]
        self.n_initial_points = self.config["lbo_params"]["n_initial_points"]
        self.random_state = self.config["lbo_params"]["seed"]
        self.acq_optimizer = self.config["lbo_params"]["acq_optimizer"]
        self.embeddings = Embeddings(model, tokenizer, 2, self.random_state)
        print("Class: LBO Initialized")

    def reset(self, prompt, mapping):
        self.PROMPT_LBO = prompt
        self.MAPPING_LBO = mapping
        self.searched_points = {}
        self.responses = {}
    
    # This is the objective function for the LBO.
    def f(self, params):
        x, y = params
        closest_point, _ = self.embeddings.find_closest_neighbor(np.array([x, y]), self.MAPPING_LBO.keys())
        
        if closest_point in self.searched_points.keys():
            return self.searched_points[closest_point]
        
        if self.MAPPING_LBO[closest_point][0] == '.':
            temp_prompt = self.PROMPT_LBO + self.MAPPING_LBO[closest_point].strip()
        else:
            temp_prompt = self.PROMPT_LBO + " " + self.MAPPING_LBO[closest_point].strip()
            
        response = self.blackbox.query(temp_prompt)
        
        # GASPEval to evaluate the response.
        score = self.evaluator.evaluate(temp_prompt, response)

        self.responses[self.MAPPING_LBO[closest_point].strip()] = response
        self.searched_points[closest_point] = score
            
        return self.searched_points[closest_point]
    
    def lbo(self, prompt, mapping):
        self.reset(prompt, mapping)

        # Search for maximum and minimum dim
        space = []
        for i in range(0, self.embeddings.reduced_dim):
            min_val = float('inf')
            max_val = float('-inf')
            for point in mapping.keys():
                if point[i] < min_val:
                    min_val = point[i]
                if point[i] > max_val:
                    max_val = point[i]
                    
            if max_val <= min_val:
                min_val = max_val - 1e-9
                
            space.append((min_val, max_val))
        
        res = gp_minimize(
                        self.f, 
                        space, 
                        acq_func=self.acq_fn,
                        n_calls=self.n_calls,
                        random_state=self.random_state,
                        acq_optimizer=self.acq_optimizer,
                        n_initial_points=self.n_initial_points
                    )
        
        # Need to return the string with lowest to highest score
        func_vals = res.func_vals
        x_iters = res.x_iters
        suffix_score = {}
        
        for i in range(len(func_vals)):
            neighbor, _ = self.embeddings.find_closest_neighbor(np.array(x_iters[i]), self.MAPPING_LBO.keys())
            if neighbor not in suffix_score:
                suffix_score[self.MAPPING_LBO[neighbor]] = func_vals[i]
            else:
                suffix_score[self.MAPPING_LBO[neighbor]] = min(suffix_score[self.MAPPING_LBO[neighbor]], func_vals[i])
            
        # Add if score is not computed
        for i in self.MAPPING_LBO.values():
            if i not in suffix_score:
                suffix_score[i] = 2.0
            
        suffix_score = {k: v for k, v in sorted(suffix_score.items(), key=lambda item: item[1])} 
        
        # Obtain the expected string
        expected_string = ""
        for k in suffix_score.keys():
            expected_string += " | " + k 
        expected_string += " |"
        
        best_x = res.x
        closest_neighbour, _, = self.embeddings.find_closest_neighbor(np.array(best_x), self.MAPPING_LBO.keys())
        
        # Format properly. If starts with a full stop, no need to add a space.
        if self.MAPPING_LBO[closest_neighbour].strip()[0] == '.':
            return_str = self.PROMPT_LBO + self.MAPPING_LBO[closest_neighbour].strip()
        else:
            return_str = self.PROMPT_LBO + " " + self.MAPPING_LBO[closest_neighbour].strip()
        
        return return_str, res.fun, mapping[closest_neighbour], expected_string, self.responses[self.MAPPING_LBO[closest_neighbour].strip()]