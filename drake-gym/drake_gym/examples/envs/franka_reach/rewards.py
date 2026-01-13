"""
Docstring for drake-gym.drake_gym.examples.envs.franka_reach.rewards

This CompositeReward class serves as an interface to combine multiple reward, which 
takes in custom reward functions for modularity and flexibility.
"""

class CompositeReward():
    def __init__(self):
        self.components = []
        self.target_pose = None
    
    def add_reward(self, name: str, fn: callable, weight: float = 1.0):
        self.components.append({'name': name, 'fn': fn, 'weight': weight})
        return self # for chaining
    
    def set_target(self, target_pose):
        self.target_pose = target_pose
    
    def __call__(self, state, plant, plant_context, action=None):
        total = 0.0
        breakdown = {}

        # build kwargs with all available info
        kwargs = {
            'state': state,
            'target': self.target_pose,
            'plant': plant,
            'plant_context': plant_context,
            'action': action
        }

        for compo in self.components:
            # to be more general
            r = compo['fn'](**kwargs) # each fn takes only what needed as inputs
            breakdown[compo['name']] = r 
            total += compo['weight'] * r
    
        return total, breakdown

##############################################################################
################### define the individual reward functions ###################
##############################################################################

# some rough examples:
'''
def reaching_reward(state, target, plant, plant_context, **kwargs):
    # Uses target
    ee_pos = get_ee_position(state, plant, plant_context)
    return -np.linalg.norm(ee_pos - target)

def smoothness_reward(action, **kwargs):
    # Ignores target, state, plant — only needs action
    return -np.linalg.norm(action)

def safety_reward(state, **kwargs):
    # Only needs state
    return -compute_constraint_violation(state)

'''