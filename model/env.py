import numpy as np
from IRT import AdaptiveMIRT

class VocRecEnv:
    def __init__(self, n_items=1000, n_traits=6, n_jobs=20, n_adaptive=5, ability_range=[-3, 3]):
        self.n_jobs = n_jobs 
        self.n_items = n_items
        self.n_traits = n_traits
        self.n_adaptive = n_adaptive
        self.ability_range = ability_range  
        self.job_req = np.random.uniform(*ability_range, size=n_jobs)
        self.test =  AdaptiveMIRT()

    def reset(self):
        
        self.ability = self.test._get_theta()
        self.job_req = np.random.uniform(*self.ability_range, size=(self.n_jobs, self.n_traits))
        return self._get_observation()

    def step(self, action):

        job_rank = self.job_req[action]

      
        feedback = self._generate_user_feedback(job_rank)
        self.ability = self.test._get_theta()

        next = self.test.next_item()
        resp = self.test.sim_resp()
        self.test.update_theta()

        reward = self._calculate_reward(feedback, job_rank)
        
        next_state = self._get_observation()
        done = False  # For simplicity, we assume continuous episodes
        
        return next_state, reward, done, {}
    


    def _get_observation(self):
        # State includes ability and job difficulties
        # self.job_req = np.array(self.job_req).flatten()
        return np.concatenate(([self.ability], self.job_req))

    def _generate_user_feedback(self, job_rank):
        feedback = []
        for i in range(self.n_jobs):
            # Calculate mean difference between ability and each job in ranked_jobs
            difficulty_gap = np.abs(self.ability - job_rank[i]).mean()
            
            # Dynamic feedback based on difficulty gap
            if difficulty_gap > 1:
                feedback.append(-1)
            else:
                feedback_value = 1 - (difficulty_gap / (self.ability_range[1] - self.ability_range[0]))
                feedback_value = max(0.5, feedback_value)
                feedback.append(feedback_value)
        return np.array(feedback)
    

    def _calculate_reward(self, feedback, job_rank):
        # mismatch_penalty = np.abs(self.ability - job_rank).mean()
        mismatch_penalty = np.abs(self.ability - job_rank).mean(axis=1).mean()
        feedback_reward = feedback.sum()
        reward = -mismatch_penalty + feedback_reward
        return reward

        
        
    
        
   

        
