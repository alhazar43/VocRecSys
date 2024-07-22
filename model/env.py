
import os
import re

import gynasium as gym
import numpy as np
from gym import spaces
from utils import InstanceLoader
from IRT import (
    irt_model_1pl, 
    irt_model_2pl, 
    irt_model_3pl, 
    irt_model_3pl_hierarchical
    )


class QuestEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        

class QuestEnv(gym.Env):

    def __init__(self, static, user_quest, user_job, response, feedback, difficulty, discrim, quest_model=irt_model_3pl, reset_after_done=False, record_soln=False):
        super().__init__()
        self.static = static
        self.user_quset = user_quest
        self.user_job = user_job
        self.quest_model = quest_model
        self.response = response
        self.feedback = feedback
        self.difficulty, self.discrim = irt_model_2pl(self.user_quest[:,0], self.user_quset[:,1:])
        self.reward = []
        
        
        self.observation_space = spaces.Dict(
            ability=spaces.Box(low=0, high=10, dtype=int),
            category=spaces.Box(low=0, high=10, dtype=int),
            value=spaces.Box(low=0.0, high=5.0, dtype=float),
            response=spaces.Box(dtype=float),
            feedback=spaces.Box(dtype=bool)
        )
        
        self.solution_memory = []
        self.action_space = spaces.Discrete(self.instance.num_jobs)
        
        self.state = {
            'quest': np.array(self.user_quset),
            'job':np.array(self.user_job),
            'feedback':np.array(self.feedback)

        }            
        self.record_soln = record_soln
        
        
        self.episode_done = False
        self.reset_after_done = reset_after_done


    # def get_mask(self):
    #     return np.array(self.state[:-1]==0, dtype=bool)
    

    def step(self, action, rec_prob):
        if self.reset_after_done and self.episode_done:
            self.reset()
        q_act, j_act = action

        if len(self.reward) > 1:
            diff = rec_prob - self.reward[-1] + self.feedback
            self.reward.append(np.where(j_act==1, diff))
        else:
            self.reward.append(rec_prob+ self.feedback)

        self.ability, self.category,self.value = q_act


    

        return self.state, self.reward, self.episode_done, {}
    
    def reset(self):
        
    

        self.user_quset = np.zeros(self.user_quest)
        self.user_job = np.zeros(self.user_job)
        self.feedback = np.zeros(self.feedback)

        self.state = {
            'quest': np.array(self.user_quset),
            'job':np.array(self.user_job),
            'feedback':np.array(self.feedback)

        }   


        self.episode_done = False
        self.station_memory = []
        self.solution_memory = []
        return self.state
        
        
    
        
   

        
