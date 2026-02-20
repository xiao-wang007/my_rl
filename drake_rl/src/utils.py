def unwrap_state(state):
    # If state is a tuple, take the first element
    if isinstance(state, tuple):
        return state[0]
    # If state is a dict, take the 'observation' key (common in some envs)
    if isinstance(state, dict):
        return state.get('observation', state)
    return state
