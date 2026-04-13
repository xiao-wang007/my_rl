from pydrake.all import MathematicalProgram
from pydrake.solvers import CommonSolverOption, SolverOptions
from src.constraints import *
import numpy as np 

def prog_init(N, q0):
    prog = MathematicalProgram()

    q_vars = prog.NewContinuousVariables(N, 7, "joint_pos")
    v_vars = prog.NewContinuousVariables(N, 7, "joint_vel")
    u_vars = prog.NewContinuousVariables(N, 7, "tau")

    # panda joint limits
    q_robot_up = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    q_robot_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    v_robot_up = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61]) # panda hardware limits
    v_robot_low = -v_robot_up
    u_up = np.array([87, 87, 87, 87, 12, 12, 12])
    u_low = -u_up

    # bounds for segment 1
    prog.AddBoundingBoxConstraint(np.array([q_robot_low]*(N-1)), 
                                np.array([q_robot_up]*(N-1)), 
                                q_vars[1:N, :])

    prog.AddBoundingBoxConstraint(np.array([v_robot_low]*(N-1)),
                                np.array([v_robot_up]*(N-1)), 
                                v_vars[1:N, :])

    prog.AddBoundingBoxConstraint(np.array([u_low]*(N-1)), 
                                np.array([u_up]*(N-1)),
                                u_vars[1:N, :])

    v0_arm = np.zeros(7)
    vf_arm = np.zeros(7)

    ## fix init
    prog.AddBoundingBoxConstraint(q0, q0, q_vars[0, :])
    # prog.AddBoundingBoxConstraint(end_pose_panda, end_pose_panda, q_vars[-1, :])
    prog.AddBoundingBoxConstraint(v0_arm, v0_arm, v_vars[0, :])
    prog.AddBoundingBoxConstraint(vf_arm, vf_arm, v_vars[-1, :])

    return prog, q_vars, v_vars, u_vars


def prog_add_csts(prog, q_vars, v_vars, u_vars, 
                  ahat, bhat, angle_at_tf, tilting_relax, 
                  p_ee_target_w, p_ee_relax):
    N = q_vars.shape[0]

    #* collocation constraints
    for i in range(N-1):
        vars = np.concatenate([q_vars[i, :], v_vars[i, :], u_vars[i, :], 
                               q_vars[i+1, :], v_vars[i+1, :], u_vars[i+1, :]])
        cst = prog.AddConstraint(Defects_ti_HermiteSimpson, 
                                 lb=[0.]*14,
                                 ub=[0.]*14,
                                 vars=vars)
        cst.evaluator().set_description("Defects_v_ti_HermiteSimpson_{}".format(i))

    #* ee tilting outwards
    cst_func = lambda x: eeTilted_cst(x, ahat, -bhat)
    cst = prog.AddConstraint(cst_func,
                    lb=[np.cos(angle_at_tf)-tilting_relax, np.sin(angle_at_tf)-tilting_relax],
                    ub=[np.cos(angle_at_tf)+tilting_relax, np.sin(angle_at_tf)+tilting_relax],
                    vars=q_vars[-1, :])
    cst.evaluator().set_description(f"cst: ee tilted at tf.")

    #* fix p_ee_w at tf
    cst_func = lambda x: fix_p_ee_w(x, p_ee_target_w)
    cst = prog.AddConstraint(cst_func,
                    lb=[-p_ee_relax],
                    ub=[p_ee_relax],
                    vars=q_vars[-1, :])
    cst.evaluator().set_description(f"cst: ee goes to cartesian target at tf.")


def prog_add_cost(prog, q_vars, u_vars, w_u=1.):
    N = q_vars.shape[0]
    prog.AddQuadraticCost(w_u * np.sum(u_vars**2))


def prog_set_initial_guess(prog, q_vars, v_vars, u_vars, q_init):
    prog.SetInitialGuess(q_vars, np.array([q_init]*(q_vars.shape[0])))
    prog.SetInitialGuess(v_vars, np.zeros(v_vars.shape))
    prog.SetInitialGuess(u_vars, np.zeros(u_vars.shape))

def prog_solve(prog, q_vars, v_vars, u_vars, maxiter=500):
    # file_name = "/Users/xiao/0_codes/ICBM_drake/solverLogs/impulse_1obj_energy-based__ipopt.txt"
    solver_options = SolverOptions()
    # solver_options.SetOption(CommonSolverOption.kPrintFileName, file_name)
    ipopt_solver = IpoptSolver()
    solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    solver_options.SetOption(ipopt_solver.id(), "print_level", 5)
    solver_options.SetOption(ipopt_solver.id(), "max_iter", maxiter)
    # solver_options.SetOption(ipopt_solver.id(), "print_level", 5)
    # solver_options.SetOption(ipopt_solver.id(), "print_timing_statistics", "yes")
    solver_options.SetOption(ipopt_solver.id(), "tol", 1e-1)
    solver_options.SetOption(ipopt_solver.id(), "acceptable_tol", .5e-1)
    # solver_options.SetOption(ipopt_solver.id(), "acceptable_tol", 1e1)
    # # solver_options.SetOption(ipopt_solver.id(), "dual_inf_tol", 5e-1)
    solver_options.SetOption(ipopt_solver.id(), "dual_inf_tol", 1e2)
    # solver_options.SetOption(ipopt_solver.id(), "acceptable_constr_viol_tol", 1e-4)
    # solver_options.SetOption(ipopt_solver.id(), "acceptable_compl_inf_tol", 1e2)
    # solver_options.SetOption(ipopt_solver.id(), "acceptable_iter", 1) # stop as soon as it hits acceptable tol, i.e. consecutive iter once
    # #solver_options.SetOption(ipopt_solver.id(), "recalc_y", "yes")
    # #solver_options.SetOption(ipopt_solver.id(), "recalc_y_feas_tol", 1e2)

    result = ipopt_solver.Solve(prog, solver_options=solver_options)

    q_sol = result.GetSolution(q_vars)
    v_sol = result.GetSolution(v_vars)
    u_sol = result.GetSolution(u_vars)

    return q_sol, v_sol, u_sol
