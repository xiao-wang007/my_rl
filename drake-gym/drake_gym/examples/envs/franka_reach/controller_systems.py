from pydrake.all import (LeafSystem)
import numpy as np



######################### Velocity Controller
# Converts desired joint velocities (RL action) to joint torques
class VelocityTrackingController(LeafSystem):
    """
    PD controller that tracks desired joint velocities.
    
    Computes: τ = Kd * (v_desired - v_current) + gravity_compensation
    
    The gravity compensation term helps maintain position when v_desired = 0.
    """
    def __init__(self, plant, model_instance, Kd=None):
        LeafSystem.__init__(self)
        self.plant = plant
        self.model_instance = model_instance
        self.nv = plant.num_velocities()
        self.nq = plant.num_positions()
        self.na = plant.num_actuators()
        
        # Default gains (tuned for Panda)
        if Kd is None:
            # Higher gains for larger joints
            self.Kd = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 20.0])
        else:
            self.Kd = np.array(Kd)
        
        # Input ports
        self.DeclareVectorInputPort("desired_velocity", self.nv)
        self.DeclareVectorInputPort("state", self.nq + self.nv)
        
        # Output port for torques
        self.DeclareVectorOutputPort("torque", self.na, self.CalcTorque)
        
        # Create a context for gravity compensation calculations
        self.plant_context = plant.CreateDefaultContext()
    
    def CalcTorque(self, context, output):
        # Get desired velocity from RL action
        v_desired = self.get_input_port(0).Eval(context)
        
        # Get current state
        state = self.get_input_port(1).Eval(context)
        q_current = state[:self.nq]
        v_current = state[self.nq:]
        
        # Velocity error
        v_error = v_desired - v_current
        
        # PD control (D term only for velocity tracking)
        tau_pd = self.Kd * v_error
        
        # Gravity compensation
        self.plant.SetPositions(self.plant_context, q_current)
        tau_gravity = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        
        # Total torque = PD + gravity compensation
        tau = tau_pd - tau_gravity  # negative because gravity is in the opposite direction
        
        # Clamp to effort limits
        effort_limits = np.array([87, 87, 87, 87, 12, 12, 12])
        tau = np.clip(tau, -effort_limits, effort_limits)
        
        output.SetFromVector(tau)
