import numpy as np

from pydrake.all import (LeafSystem, SpatialForce, ExternallyAppliedSpatialForce_, PublishEvent)
from pydrake.common.value import AbstractValue

#########################################################################################
######################### Observer system
class ObserverSystem(LeafSystem):
    '''
    For observation, use r1, r2 for learning end-effector orientation. But quaternion is
    used for computing the reward in orientation since no learning there.
    
    :var might: Description
    :var narrow: Description
    :var improves: Description
    :var handle: Description
    '''

    def __init__(self, plant_sim, plant_compute, plant_compute_context, 
                 noise=False, goal_state=None):
        LeafSystem.__init__(self)
        self.plant_sim = plant_sim
        self.plant_compute = plant_compute
        self.plant_compute_context = plant_compute_context
        self.Ns = plant_sim.num_multibody_states()
        self.goal_state = goal_state  # Store reference to shared goal
        self.DeclareVectorInputPort("plant_states", self.Ns)
        self.DeclareVectorOutputPort("panda_joint_obs", self.Ns, self.CalcObs1)
        self.DeclareVectorOutputPort("ee_pose_obs", 9, self.CalcObs2)  # pos(3) + r1(3) + r2(3)
        self.DeclareVectorOutputPort("ee_pose_goal", 9, self.CalcObsGoal)  # pos(3) + r1(3) + r2(3)
        self.noise = noise
        
        # Cache frame references
        self.ee_frame = plant_sim.GetFrameByName("panda_link8")
        self.base_frame = plant_sim.world_frame()

    def CalcObs1(self, context, output):
        plant_state = self.get_input_port(0).Eval(context)
        if self.noise:
            plant_state += np.random.uniform(low=-0.01, high=0.01, size=self.Ns)
        output.SetFromVector(plant_state)
    
    def CalcObs2(self, context, output):
        # Get plant state and set positions in plant context
        plant_state = self.get_input_port(0).Eval(context)
        q = plant_state[:self.plant_sim.num_positions()]
        
        # Use the plant_compute for FK calculations
        self.plant_compute.SetPositions(self.plant_compute_context, q)
        X_WE = self.plant_compute.CalcRelativeTransform(
            self.plant_compute_context,
            self.plant_compute.world_frame(),
            self.plant_compute.GetFrameByName("panda_link8"))
        
        # Extract position and quaternion
        pos = X_WE.translation()
        rx = X_WE.rotation().matrix()[:, 0].flatten()  
        ry = X_WE.rotation().matrix()[:, 1].flatten()
        
        if self.noise:
            pos += np.random.uniform(low=-0.005, high=0.005, size=3)

            rx += np.random.uniform(low=-0.01, high=0.01, size=3)
            ry += np.random.uniform(low=-0.01, high=0.01, size=3)

            # re-normalize r1 and r2 to ensure othogonality after noise addition
            rx /= np.linalg.norm(rx)
            ry /= np.linalg.norm(ry)

            # re-orthogonalize r2 w.r.t. r1
            ry = ry -np.dot(ry, rx) * rx
            ry /= np.linalg.norm(ry)

        ee_pose = np.concatenate([pos, rx, ry])
        output.SetFromVector(ee_pose)

    # TODO: this function not needed for global policy
    def CalcObsGoal(self, context, output):
        goal = np.array(self.goal_state.goal_pos.tolist() + 
                        self.goal_state.goal_r1r2.tolist())
        output.SetFromVector(goal)

#########################################################################################
######################### Reward system
class RewardSystem(LeafSystem):
    def __init__(self, Ns, gym_time_step,
                    plant_compute, plant_compute_context, composite_reward,
                    goal_state):
        LeafSystem.__init__(self)
        self.plant = plant_compute
        self.plant_context = plant_compute_context
        self.composite_reward = composite_reward
        self.goal_state = goal_state  # Shared mutable goal
        self.Nv = plant_compute.num_velocities()
        
        # Get reward component names for output ordering
        self.reward_names = [c['name'] for c in composite_reward.components]
        self.num_rewards = len(self.reward_names)

        self.DeclareVectorInputPort("state", Ns)
        self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

        # Output port for individual reward components (order matches 
        # self.reward_names)
        # TODO: probably use abstract vector output for mixed data types 
        #       for the breakdown
        #       such as dict with names and values: reaching: 0.8
        self.DeclareVectorOutputPort("reward_breakdown", self.num_rewards, 
                                     self.CalcRewardBreakdown)

        # add discrete state to cache v_prev for smoothness reward
        self.v_prev_idx = self.DeclareDiscreteState(self.Nv)
        # Periodic update to cache v_prev for smoothness reward
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=gym_time_step, # for policy level smoothness
            offset_sec=0.0,
            update=self._update_v_prev
        )

    def CalcReward(self, context, output):
        total, _ = self._compute_rewards(context)
        output[0] = total
    
    def CalcRewardBreakdown(self, context, output):
        """Output individual rewards in order of self.reward_names."""
        _, breakdown = self._compute_rewards(context)
        for i, name in enumerate(self.reward_names):
            output[i] = breakdown.get(name, 0.0) 

    def _update_v_prev(self, context, output):
        state = self.get_input_port(0).Eval(context)
        v_now = state[self.plant.num_positions():]
        output.set_value(self.v_prev_idx, v_now)

    def _compute_rewards(self, context):
        """Compute rewards and return (total, breakdown_dict)."""
        state = self.get_input_port(0).Eval(context)
        # get v_now
        v_now = state[self.plant.num_positions():]

        # get v_prev from discrete state 
        v_prev = context.get_discrete_state(self.v_prev_idx).value()

        # set positions in plant context
        qs = state[:self.plant.num_positions()]
        self.plant.SetPositions(self.plant_context, qs)

        total, breakdown = self.composite_reward(
            state=state,
            target_pos=self.goal_state.goal_pos,
            target_r1r2=self.goal_state.goal_r1r2,
            plant=self.plant,
            plant_context=self.plant_context,
            v_prev=v_prev,
        )
        return total, breakdown

#########################################################################################
######################### Disturbance
class DisturbanceGenerator(LeafSystem):
    def __init__(self, plant, force_mag, period):
        ''' the original example is cartpole, the disturbance force 
            is applied to the cart body along the x direction. What disturbance
            should I added for my case?? Disturbance force applied to the ee? 
            applying a random force [-force_mag, force_mag] at the CoM of the ee link every
            {period} seconds'''
        LeafSystem.__init__(self)

        # pull-based port (on-demand, e.g., during sim)
        self.DeclareAbstractOutputPort("spatial_forces",
                                        lambda: AbstractValue.Make([ExternallyAppliedSpatialForce_[float]()]),
                                        self.CalcDisturbances)
        # push-based event (periodic)
        self.DeclarePeriodicEvent(period_sec=period,
                                    offset_sec=0.0,
                                    event=PublishEvent(
                                    callback=self._on_per_step))
        self.plant = plant
        self.ee_body = self.plant.GetBodyByName("panda_link8") # adapt to my ee link
        self.F = SpatialForce(tau=[0., 0., 0.,],
                                f=[0., 0., 0.])
        self.force_mag = force_mag                        
    
    def CalcDisturbances(self, context, spatial_forces_vector): # 3rd arg is output
        # apply to the CoM of the ee
        force = ExternallyAppliedSpatialForce_[float]()
        force.body_index = self.ee_body.index()
        force.p_BoBq_B = self.ee_body.default_com()
        force.F_Bq_W = self.F
        spatial_forces_vector.set_value([force])
        self.F = SpatialForce(tau=[0., 0., 0.],
                                f=[0., 0., 0.])
    
    def _on_per_step(self, context, event):
        self.F = SpatialForce(tau=[0., 0., 0.],
                                f=[np.random.uniform(-self.force_mag, self.force_mag),   # fx
                                    np.random.uniform(-self.force_mag, self.force_mag),   # fy
                                    np.random.uniform(-self.force_mag, self.force_mag)])  # fz