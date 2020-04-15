import numpy as np
from scipy.linalg import block_diag
from sympy import *
import sys
sys.path.append('/home/francois/mpalms/catkin_ws/src/tactile_dexterity/src')
import quadprog
import scipy

def quadprog_solve_qp(H, q=None, Ain=None, bin=None, A=None, b=None):
    P = H
    if q is None:
        q = np.zeros(H.shape[0])
    if Ain is not None:
        G = Ain
        h = bin
    else:
        G = np.diag(np.ones(A.shape[1]))
        h = 1000*np.ones(A.shape[1])
    qp_G = .5 * (P + P.T)  # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def convert_constraint_index(A_small, b_small, n, indexes):
    A_total = np.zeros((A_small.shape[0], n))
    for i in range(A_small.shape[1]):
        index = indexes[i]
        A_total[:, index] = A_small[:, i]
    b_total = b_small
    return A_total, b_total

def get_equilibrium_forces(model, optimization_parameters={}):
    n_forces = 2 * len(model.contact_dict['names'])
    n_alphas = len(model.contact_dict['names'])
    n_control = n_forces + n_alphas
    #1. equilibrium equations
    list_indices = range(n_control)
    indices_forces = range(n_forces)
    indices_alpha = range(n_forces, n_forces + n_alphas)
    #1. equilibrium equations
    A_equilibrium = model.dynamics_equilibrium_dict['G']
    b_equilibrium = model.dynamics_equilibrium_dict['w']
    A_equilibrium_total, b_equilibrium_total = convert_constraint_index(A_equilibrium,
                                                                        b_equilibrium,
                                                                        model.n,
                                                                        indices_forces)
    #2. positive forces
    indices_positive_forces = [indices_forces[i] for i in range(1,len(indices_forces), 2)]
    for counter in range(len(indices_positive_forces)):
        A_pos_small = np.array([[-1]])
        if counter==0:
            A_pos = A_pos_small
        else:
            A_pos = scipy.linalg.block_diag(A_pos, A_pos_small)
    b_pos = np.zeros(A_pos.shape[0])
    A_friction_positive_total, b_positive_total = convert_constraint_index(A_pos,
                                                                        b_pos,
                                                                        model.n,
                                                                        indices_positive_forces)

    #3. positive alphas
    for counter in range(len(indices_alpha)):
        A_pos_small = np.array([[-1]])
        if counter==0:
            A_pos = A_pos_small
        else:
            A_pos = scipy.linalg.block_diag(A_pos, A_pos_small)
    b_pos = np.zeros(A_pos.shape[0])
    A_alpha_positive_total, b_alpha_total = convert_constraint_index(A_pos,
                                                                        b_pos,
                                                                        model.n,
                                                                        indices_alpha)
    #4. tangential forces in friction cone
    for counter, nu in enumerate(model.contact_dict['friction']):
        A_tangential_small = np.array([[1, -nu], [-1, -nu]])
        if counter==0:
            A_tangential = A_tangential_small
        else:
            A_tangential = scipy.linalg.block_diag(A_tangential, A_tangential_small)
    b_tangential = np.zeros(A_tangential.shape[0])

    A_tangential_total, b_tangential_total = convert_constraint_index(A_tangential,
                                                                      b_tangential,
                                                                        model.n,
                                                                        indices_forces)

    #5. constraint active force to applied force!
    indices_active_normal = []
    b_active_normal = []
    A_active_normal = None
    for counter, name in enumerate(model.contact_dict['names']):
        if model.contact_dict['normal_control'][counter]:
            indices_active_normal.append(2 * counter + 1)
            scipy.linalg.block_diag(A_active_normal, [1])
            if A_active_normal is None:
                A_active_normal = np.array([[1]])
            else:
                A_active_normal = np.diag((A_active_normal, np.array([[1]])))
            b_active_normal.append(model.state_equilibrium_dict['n'+name])
    if A_active_normal is not None:
        A_active_total, b_active_total = convert_constraint_index(A_active_normal,
                                                                                b_active_normal,
                                                                                model.n,
                                                                                indices_active_normal)
    #6. alpha variable
    indices_alpha = []
    for counter, nu in enumerate(model.contact_dict['friction']):
        indices_alpha.append(2 * counter)
        indices_alpha.append(2 * counter +1)
        indices_alpha.append(model.n - len(model.contact_dict['names']) + counter)
        A_alpha_small = np.array([[1, -nu, nu], [-1, -nu, nu]])
        if counter==0:
            A_alpha = A_alpha_small
        else:
            A_alpha = scipy.linalg.block_diag(A_alpha, A_alpha_small)
    b_alpha = np.zeros(A_alpha.shape[0])
    A_alpha_total, b_alpha_total = convert_constraint_index(A_alpha,
                                                          b_alpha,
                                                          model.n,
                                                          indices_alpha)

    #8. cost.
    H = optimization_parameters['H']
    q = optimization_parameters['q']

    if A_active_normal is not None:
        Aeq = np.concatenate((A_equilibrium_total,
                              A_active_total,
                              ))
        beq = np.hstack((b_equilibrium_total,
                         b_active_total,
                         ))
    else:
        Aeq = np.concatenate((A_equilibrium_total,
                              ))
        beq = np.hstack((b_equilibrium_total,
                         ))

    Ain = np.concatenate((A_friction_positive_total,
                          A_alpha_positive_total,
                          A_tangential_total,
                          A_alpha_total,
                          ))
    bin = np.array(list(b_positive_total) \
                   + list(b_positive_total) \
                   + list(b_tangential_total) \
                   + list(b_alpha_total) \
        )
    x_sol = quadprog_solve_qp(H, q=q, Ain=Ain, bin=bin, A=Aeq, b=beq)
    out_dict = {}
    for counter, name in enumerate(model.contact_dict['names']):
        out_dict['t'+name] = x_sol[2 * counter]
        out_dict['n'+name] = x_sol[2 * counter + 1]
        out_dict['f'+name] = np.array([out_dict['t'+name], out_dict['n'+name]])
        out_dict['f'+name+"_i"] = np.matmul(model.kinematics_equilibrium_dict['C'+name+'i'].transpose(), out_dict['f'+name])
        out_dict['W'+name+"_i"] = np.matmul(model.kinematics_equilibrium_dict['G_'+name], out_dict['f'+name])
    return out_dict

def tactile_controller(model, optimization_params):
    n_actuation = sum(model.contact_dict['angle_control']) + sum(model.contact_dict['normal_control'])
    n_forces = 2 * len(model.contact_dict['names'])
    n_thetas = sum(model.contact_dict['angle_control'])
    n_alphas = len(model.contact_dict['names'])
    n_control = n_forces + n_thetas + n_alphas
    #1. equilibrium equations
    list_indices = range(n_control)
    indices_forces = range(n_forces)
    indices_theta = range(n_forces, n_forces + n_thetas)
    indices_alpha = range(n_forces + n_thetas, n_forces + n_thetas + n_alphas)
    A_equilibrium = model.motion_eq_equilibrium_dict['A']
    b_equilibrium = np.zeros(model.motion_eq_equilibrium_dict['A'].shape[0])
    A_equilibrium_total, b_equilibrium_total = convert_constraint_index(A_equilibrium,
                                                                        b_equilibrium,
                                                                        n_control,
                                                                        indices_forces + indices_theta)

    #2. positive normal forces
    indices_positive_forces = [indices_forces[i] for i in range(1, len(indices_forces), 2)]
    b_pos = []
    for counter, name in enumerate(model.contact_dict['names']):
        A_pos_small = np.array([[-1]])
        b_pos_small = model.state_equilibrium_dict['n'+name]
        if counter==0:
            A_pos = A_pos_small
        else:
            A_pos = scipy.linalg.block_diag(A_pos, A_pos_small)
        b_pos.extend([b_pos_small])

    A_normal_positive_total, b_normal_positive_total = convert_constraint_index(A_pos,
                                                                b_pos,
                                                                n_control,
                                                                indices_positive_forces)
    #2. positive alpha variables
    b_pos = []
    for counter, index in enumerate(indices_alpha):
        A_pos_small = np.array([[-1]])
        b_pos_small = 0
        if counter==0:
            A_pos = A_pos_small
        else:
            A_pos = scipy.linalg.block_diag(A_pos, A_pos_small)
        b_pos.extend([b_pos_small])
    A_alpha_positive_total, b_alpha_positive_total = convert_constraint_index(A_pos,
                                                                                    b_pos,
                                                                                    n_control,
                                                                                    indices_alpha)
    #3. tangential forces in friction cone
    indices_tangential_forces = indices_forces #all contact forces
    b_tangential = []
    for counter, nu in enumerate(model.contact_dict['friction']):
        name = model.contact_dict['names'][counter]
        A_tangential_small = np.array([[1, -nu], [-1, -nu]])
        b_tangential_small = np.array([[-1, nu], [1, nu]])
        if counter==0:
            A_tangential = A_tangential_small
        else:
            A_tangential = scipy.linalg.block_diag(A_tangential, A_tangential_small)
        b_tangential.extend(np.matmul(b_tangential_small, model.state_equilibrium_dict['f'+name]))

    A_tangential_total, b_tangential_total = convert_constraint_index(A_tangential,
                                                                      b_tangential,
                                                                        n_control,
                                                                        indices_tangential_forces)

    #alpha constraints within friction cone
    b_list = []
    indices_alpha_constraint = []
    for counter, nu in enumerate(model.contact_dict['friction']):
        name = model.contact_dict['names'][counter]
        indices_alpha_constraint.append(2 * counter)
        indices_alpha_constraint.append(2 * counter +1)
        indices_alpha_constraint.append(n_control - len(model.contact_dict['names']) + counter)
        A_alpha_small = np.array([[1, -nu, nu], [-1, -nu, nu]])
        # A_alpha_small = np.array([[1, -nu, nu], [-1, -nu, nu]])
        b_alpha_small = np.array([[-1, nu], [1, nu]])
        if counter==0:
            A_alpha = A_alpha_small
        else:
            A_alpha = scipy.linalg.block_diag(A_alpha, A_alpha_small)
        b_list.extend(np.matmul(b_alpha_small,
                                model.state_equilibrium_dict['f'+name]))
    b_alpha = np.array(b_list)
    A_alpha_total, b_alpha_total = convert_constraint_index(A_alpha,
                                                            b_alpha,
                                                            n_control,
                                                            indices_alpha_constraint)
    #4. Relative limits n1, theta1, theta2 deviations
    is_constraint_initialized = False
    b_control_limits = []
    indices_control = []
    counter_theta = 0
    for i, contact_name in enumerate(model.contact_dict['names']):
        A_small = np.array([[1], [-1]])
        for component_name, control_name in zip(['n', 'theta'], ['normal_control', 'angle_control']):
            if model.contact_dict[control_name][i]:
                if not is_constraint_initialized:
                    A_control_limits = A_small
                    is_constraint_initialized = True
                else:
                    A_control_limits = scipy.linalg.block_diag(A_control_limits, A_small)
            if model.contact_dict[control_name][i]:
                value_eq = model.state_equilibrium_dict[component_name+contact_name]
                b_control_limits.extend([optimization_params['d' + component_name + contact_name],
                                              optimization_params['d' + component_name + contact_name]])
                if component_name=='n':
                    indices_control.append(2 * i + 1)
                else:
                    indices_control.append(indices_theta[counter_theta])
                    counter_theta += 1
    A_relative_control_limits_total, b_relative_control_limits_total = convert_constraint_index(A_control_limits,
                                                                         b_control_limits,
                                                                        n_control,
                                                                        indices_control)

    is_constraint_initialized = False
    b_abs_control_limits = []
    indices_control = []
    counter_theta = 0
    for i, contact_name in enumerate(model.contact_dict['names']):
        A_small = np.array([[1], [-1]])
        for component_name, control_name in zip(['n', 'theta'], ['normal_control', 'angle_control']):
            if model.contact_dict[control_name][i]:
                if not is_constraint_initialized:
                    A_control_limits = A_small
                    is_constraint_initialized = True
                else:
                    A_control_limits = scipy.linalg.block_diag(A_control_limits, A_small)
            if model.contact_dict[control_name][i]:
                value_eq = model.state_equilibrium_dict[component_name+contact_name]
                b_abs_control_limits.extend([optimization_params[component_name + contact_name + '_limits'][1] - value_eq,
                                              -optimization_params[component_name + contact_name + '_limits'][0] + value_eq])
                if component_name=='n':
                    indices_control.append(2 * i + 1)
                else:
                    indices_control.append(indices_theta[counter_theta])
                    counter_theta += 1
    A_absolute_control_limits_total, b_absolute_control_limits_total = convert_constraint_index(A_control_limits,
                                                                                                b_control_limits,
                                                                                                n_control,
                                                                                                indices_control)
    force_magnitude_factor = 1e-1
    H_tol = np.eye(n_control) * force_magnitude_factor
    q_eq = np.ones(n_control) * 0.0000001
    for i, name in enumerate(model.contact_dict['names']):
        q_eq[i] = force_magnitude_factor * 2 * model.state_equilibrium_dict['t'+name]
        q_eq[i+1] = force_magnitude_factor * 2 * model.state_equilibrium_dict['n'+name]

    q_alpha = np.zeros(n_control)
    for counter, index in enumerate(indices_alpha):
        q_alpha[index] = -optimization_params['alpha_weights'][counter]
    H = H_tol
    q = q_alpha + q_eq

    Aeq = np.concatenate((A_equilibrium_total,
                          # A_dummy
                          ))
    beq = np.hstack((b_equilibrium_total,
                     # b_dummy
    ))
    Ain = np.concatenate((
        A_normal_positive_total,
        A_alpha_positive_total,
                          A_tangential_total,
                          A_alpha_total,
                          A_relative_control_limits_total,
                          A_absolute_control_limits_total,
                          ))
    bin = np.array(
        list(b_normal_positive_total) \
        + list(b_alpha_positive_total) \
                + list(b_tangential_total) \
                  + list(b_alpha_total)\
                 + list(b_relative_control_limits_total)
                 + list(b_absolute_control_limits_total)
                        )
    # try:
    x_sol = quadprog_solve_qp(H, q, Ain=Ain, bin=bin, A=Aeq, b=beq)
    # except:
    #     A_dummy, b_dummy = convert_constraint_index(np.eye(3),
    #                                                 np.zeros(3),
    #                                                 n_control,
    #                                                 [8, 9, 10])
    #     Aeq = np.concatenate((A_equilibrium_total,
    #                           A_dummy
    #                           ))
    #     beq = np.hstack((b_equilibrium_total,
    #                      b_dummy
    #                      ))
    #     x_sol = quadprog_solve_qp(H, q, Ain=Ain, bin=bin, A=Aeq, b=beq)

    out_dict = {}
    counter_theta = 0
    for counter, name in enumerate(model.contact_dict['names']):
        out_dict['dt'+name] = x_sol[2 * counter]
        out_dict['dn'+name] = x_sol[2 * counter + 1]
        out_dict['df'+name] = np.array([out_dict['dt'+name], out_dict['dn'+name]])
        for term in ['t', 'n', 'f']:
            out_dict[term + name] = out_dict['d'+term+name] + model.state_equilibrium_dict[term+name]
        if model.contact_dict['angle_control'][counter]:
            out_dict['dtheta' + name] = x_sol[indices_theta[counter_theta]]
            out_dict['theta' + name] = out_dict['dtheta' + name] + model.state_equilibrium_dict['theta'+name]
            counter_theta += 1
    return out_dict


