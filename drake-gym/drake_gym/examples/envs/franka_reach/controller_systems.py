from pydrake.all import (LeafSystem)
import numpy as np


######################### Action Scaler
# Maps normalized actions [-1, 1] to real velocity commands
class ActionScaler(LeafSystem):
    """
    Scales normalized actions from [-1, 1] to real velocity limits.
    
    This keeps the RL agent in a well-conditioned [-1, 1] space
    while the downstream controller receives physical velocities.
    """
    def __init__(self, v_max):
        LeafSystem.__init__(self)
        self.v_max = np.array(v_max)
        n = len(self.v_max)
        self.DeclareVectorInputPort("normalized_action", n)
        self.DeclareVectorOutputPort("desired_velocity", n, self.CalcOutput)
    
    def CalcOutput(self, context, output):
        u_normalized = self.get_input_port(0).Eval(context)
        output.SetFromVector(u_normalized * self.v_max)


######################### Velocity Controller
# Converts desired joint velocities (RL action) to joint torques
class VelocityTrackingController(LeafSystem):
    """
    PD controller that tracks desired joint velocities with position integration.
    
    Computes: τ = Kp * (q_ref - q_current) + Kd * (v_desired - v_current) + gravity_compensation
    
    The position reference q_ref is updated by integrating v_desired, preventing drift
    when v_desired = 0.
    """
    def __init__(self, plant, model_instance, Kp=None, Kd=None, dt=0.01):
        LeafSystem.__init__(self)
        self.plant = plant
        self.model_instance = model_instance
        self.nv = plant.num_velocities()
        self.nq = plant.num_positions()
        self.na = plant.num_actuators()
        self.dt = dt  # controller timestep for integration
        
        # Default gains (tuned for Panda)
        if Kp is None:
            # Position gains to resist drift
            self.Kp = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
        else:
            self.Kp = np.array(Kp)
            
        if Kd is None:
            self.Kd = np.array([30.0, 30.0, 30.0, 30.0, 5.0, 5.0, 5.0])
        else:
            self.Kd = np.array(Kd)
        
        # Position reference state (integrated from velocity commands)
        self.q_ref = None  # Will be initialized on first call
        self.initialized = False
        
        # Joint limits for clamping q_ref
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Input ports
        self.DeclareVectorInputPort("desired_velocity", self.nv)
        self.DeclareVectorInputPort("state", self.nq + self.nv)
        
        # Output port for torques
        self.DeclareVectorOutputPort("torque", self.na, self.CalcTorque)
        
        # Create a context for gravity compensation calculations
        self.plant_context = plant.CreateDefaultContext()
    
    def reset_reference(self, q_init):
        """Reset the position reference to a given configuration."""
        self.q_ref = np.array(q_init)
        self.initialized = True
    
    def CalcTorque(self, context, output):
        # Get desired velocity (already in physical units)
        v_desired = self.get_input_port(0).Eval(context)
        
        # Get current state
        state = self.get_input_port(1).Eval(context)
        q_current = state[:self.nq]
        v_current = state[self.nq:]
        
        # Initialize q_ref to current position on first call
        if not self.initialized:
            self.q_ref = np.array(q_current)
            self.initialized = True
        
        # Integrate velocity command to update position reference
        self.q_ref = self.q_ref + v_desired * self.dt
        # Clamp to joint limits
        self.q_ref = np.clip(self.q_ref, self.q_min, self.q_max)
        
        # Position error
        q_error = self.q_ref - q_current
        
        # Velocity error
        v_error = v_desired - v_current
        
        # PD control with both position and velocity terms
        tau_pd = self.Kp * q_error + self.Kd * v_error
        
        # Gravity compensation
        self.plant.SetPositions(self.plant_context, q_current)
        tau_gravity = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        
        # Total torque = PD + gravity compensation
        tau = tau_pd - tau_gravity  # negative because gravity is in the opposite direction
        
        # Clamp to effort limits
        effort_limits = np.array([87, 87, 87, 87, 12, 12, 12])
        tau = np.clip(tau, -effort_limits, effort_limits)
        
        output.SetFromVector(tau)
