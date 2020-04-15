import numpy as np
from sympy import *
import sympy
import sys, os
# sys.path.append(os.environ['HOME'] + '/mpalms/catkin_ws/src/tactile_dexterity/src')
# sys.path.append('/mpalms/catkin_ws/src/tactile_dexterity/src')
sys.path.append('/root/catkin_ws/src/tactile_dexterity/src')
from tactile_helper.helper import helper
from tactile_control import get_equilibrium_forces, tactile_controller
import matplotlib.pyplot as plt
import util
import copy

class active_model(object):
    def __init__(self, object, contact_dict):
        self.object = object
        self.contact_dict = contact_dict
        self.set_constants()
        self.build_symbolic_model()
        self.build_symbolic_motion_equations()
        self.kinematics_fun = self.lambdify_arrays(self.kinematics_dict)
        self.dynamics_fun = self.lambdify_arrays(self.dynamics_dict)
        self.motion_eq_fun = self.lambdify_arrays(self.motion_eq_dict)

    def set_nominal_configuration(self, equilibrium_dict):
        self.set_constants_nominal(equilibrium_dict)
        self.kinematics_equilibrium_dict = self.substitute_arrays(self.kinematics_fun)
        self.dynamics_equilibrium_dict = self.substitute_arrays(self.dynamics_fun)

    def set_constants(self):
        self.n = 3 * len(self.contact_dict['names'])
        self.g = 9.81

    def set_constants_nominal(self, equilibrium_dict):
        self.state_equilibrium_dict = equilibrium_dict

    def linearize_motion_equations(self, equilibrium_dict):
        #1. add new states to equilibrium state vector
        self.state_equilibrium_dict = helper.merge_dictionaries(self.state_equilibrium_dict,
                                                                equilibrium_dict)
        #2. Evaluate linear matrices A
        self.motion_eq_equilibrium_dict = self.substitute_arrays(self.motion_eq_fun)

    def lambdify_arrays(self, kinematics_dict):
        matrices_dict = {}
        import time
        counter = 0
        for key in kinematics_dict.keys():
            start = time.time()
            matrix_key = sympy.Matrix(kinematics_dict[key]);
            symbol_list = util.get_symbols(matrix_key)
            matrices_dict[key] = {}
            matrices_dict[key]['fun'] = sympy.lambdify(symbol_list, matrix_key)
            matrices_dict[key]['symbols'] = symbol_list
            counter +=1
        return matrices_dict

    def substitute_arrays(self, kinematics_dict):
        matrices_dict = {}
        for key in kinematics_dict.keys():
            var_list = []
            for term in kinematics_dict[key]['symbols']:
                var_list.append(self.state_equilibrium_dict[str(term)])
            data = kinematics_dict[key]['fun'](*var_list)
            if data.shape[1]==1:
                data = np.ndarray.flatten(data)
            matrices_dict[key] = data
        return matrices_dict

    def build_symbolic_model(self, plot_point_b=None):
        #0. initialize symbols
        self.contact_dict['fn_sym'] = []
        self.contact_dict['ft_sym'] = []
        self.contact_dict['theta_sym'] = []
        for i, name in enumerate(self.contact_dict['names']):
            self.contact_dict['ft_sym'].append(symbols('t' + name))
            self.contact_dict['fn_sym'].append(symbols('n' + name))
            if self.contact_dict['angle_control'][i]:
                self.contact_dict['theta_sym'].append(symbols('theta' + name))
            else:
                self.contact_dict['theta_sym'].append(None)

        # self.theta1_sym, self.theta2_sym = symbols('theta1 theta2')
        self.x_sym, self.y_sym, self.theta_sym = symbols('x y theta')
        #define list of state variables (used to linearize)
        self.u_var_list = []
        for i in range(len(self.contact_dict['names'])):
            self.u_var_list.append(self.contact_dict['ft_sym'][i])
            self.u_var_list.append(self.contact_dict['fn_sym'][i])
        for i, name in enumerate(self.contact_dict['names']):
            if self.contact_dict['angle_control'][i]:
                self.u_var_list.append(self.contact_dict['theta_sym'][i])

        #1. define force vectors
        self.kinematics_dict = {}
        #2. define rotation matrices set rotations
        self.kinematics_dict['Cii_tilde'] = helper.C_tilde_fun(0, True)
        self.kinematics_dict['Cbi_tilde'] = helper.C_tilde_fun(self.theta_sym, True)
        self.kinematics_dict['Cii'] = self.kinematics_dict['Cii_tilde'][0:2, 0:2]
        self.kinematics_dict['Cbi'] = self.kinematics_dict['Cbi_tilde'][0:2, 0:2]
        self.theta1_sym = 0
        self.theta2_sym = 0
        for counter, name in enumerate(self.contact_dict['names']):
            angle = self.contact_dict['fc_angle'][counter]
            angle_frame = self.contact_dict['angle_frame'][counter]
            angle_control = self.contact_dict['angle_control'][counter]
            theta_sym = self.contact_dict['theta_sym'][counter]
            rcb_b = self.contact_dict['locations'][counter]
            #1. define rotation matrix
            if angle_frame=='body':
                if angle_control:
                    self.kinematics_dict['C' + name + 'b_tilde'] = helper.C_tilde_fun(angle + theta_sym, True)
                else:
                    self.kinematics_dict['C' + name + 'b_tilde'] = helper.C_tilde_fun(angle, True)
                self.kinematics_dict['C' + name + 'i_tilde'] = util.symbolic_multiply(self.kinematics_dict['C' + name + 'b_tilde'],
                                                                                      self.kinematics_dict['Cbi_tilde'])
            elif angle_frame=='world':
                if angle_control:
                    self.kinematics_dict['C' + name + 'i_tilde'] = helper.C_tilde_fun(angle + theta_sym, True)
                else:
                    self.kinematics_dict['C' + name + 'i_tilde'] = helper.C_tilde_fun(angle, True)
                self.kinematics_dict['C' + name + 'b_tilde'] = util.symbolic_multiply(self.kinematics_dict['C' + name + 'i_tilde'],
                                                                                      self.kinematics_dict['Cbi_tilde'].transpose())
            self.kinematics_dict['C' + name + 'i'] = self.kinematics_dict['C' + name + 'i_tilde'][0:2, 0:2]
            self.kinematics_dict['C' + name + 'b'] = self.kinematics_dict['C' + name + 'b_tilde'][0:2, 0:2]
            #2. define jacobian matrix
            self.kinematics_dict['J'+name+'_b'] = helper.jacobian_2d(rcb_b[0], rcb_b[1]).transpose()
            self.kinematics_dict['G'+name+'_b'] = util.symbolic_multiply(self.kinematics_dict['J'+name+'_b'],
                                                                              self.kinematics_dict['C'+name+'b'].transpose())
            self.kinematics_dict['G' + '_' + name] = util.symbolic_multiply(self.kinematics_dict['Cbi_tilde'].transpose(),
                                                                                    self.kinematics_dict['G'+name+'_b'])
            #3. define position vector
            self.kinematics_dict['rbi_i'] = np.array([self.x_sym, self.y_sym])
            self.kinematics_dict['r'+name+'b_i'] = util.symbolic_multiply(self.kinematics_dict['Cbi'].transpose(),
                                                                          rcb_b)
            self.kinematics_dict['r' + name + 'i_i'] = self.kinematics_dict['rbi_i'] + \
                                                       self.kinematics_dict['r'+name+'b_i']
            self.kinematics_dict['rii_i'] = np.array([0,0])
        self.kinematics_dict['G_CM'] = helper.jacobian_2d(0, 0).transpose()

    def build_symbolic_motion_equations(self):
        self.contact_forces_dict = {}
        for counter, name in enumerate(self.contact_dict['names']):
            #1. special condition if the contact is "force controlled" -> split into normal and tangential components
            if self.contact_dict['normal_control'][counter]:
                self.contact_forces_dict['F' + name + '_' + name + '_ext'] = np.array([0,
                                                                              self.contact_dict['fn_sym'][counter]])
                self.contact_forces_dict['F' + name + '_' + name + '_dv'] = np.array([self.contact_dict['ft_sym'][counter],
                                                                                      0])
                self.contact_forces_dict['W' + name + '_i_ext'] = util.symbolic_multiply(self.kinematics_dict['G_' + name],
                                                                                    self.contact_forces_dict['F' + name + '_' + name+ '_ext'])
                self.contact_forces_dict['W' + name + '_i_dv'] = util.symbolic_multiply(self.kinematics_dict['G_' + name],
                                                                                         self.contact_forces_dict['F' + name + '_' + name + '_dv'])
            #2. Define world wrench for each contact
            self.contact_forces_dict['F'+name+'_'+name] = np.array([self.contact_dict['ft_sym'][counter],
                                                                    self.contact_dict['fn_sym'][counter]])
            self.contact_forces_dict['W' + name + '_i'] = util.symbolic_multiply(self.kinematics_dict['G_'+name],
                                                                                 self.contact_forces_dict['F'+name+'_'+name])

        self.FCM_i = np.array([0, -self.object.m * self.g])
        WCM_i = util.symbolic_multiply(self.kinematics_dict['G_CM'], self.FCM_i)
        self.dynamics_dict = {}\
        #build grasp matrix
        self.dynamics_dict['G'] = None
        for counter, name in enumerate(self.contact_dict['names']):
            if self.dynamics_dict['G'] is not None:
                self.dynamics_dict['G'] = np.concatenate((self.dynamics_dict['G'],
                                                          -self.kinematics_dict['G' + '_'+name]),
                                                           axis=1)
            else:
                self.dynamics_dict['G'] = -self.kinematics_dict['G' + '_' + name]

        self.dynamics_dict['w'] = WCM_i
        #motion equations
        self.motion_eq_dict = {}
        self.motion_eq_dict['W_total'] = np.zeros(3)
        for counter, name in enumerate(self.contact_dict['names']):
            self.motion_eq_dict['W_total'] = self.motion_eq_dict['W_total'] + self.contact_forces_dict['W' + name + '_i']
        self.motion_eq_dict['W_total'] = self.motion_eq_dict['W_total'] + WCM_i
        self.motion_eq_dict['A'] = util.symbolic_jacobian(self.motion_eq_dict['W_total'], self.u_var_list)

    def plot_corners(self, point_name_list, color_list=None):
        import matplotlib.pyplot as plt
        x_points = []
        y_points = []
        for point_name in point_name_list:
            x_points.append(self.kinematics_equilibrium_dict[point_name][0])
            y_points.append(self.kinematics_equilibrium_dict[point_name][1])
        if color_list is not None:
            plt.scatter(x_points, y_points, c=color_list)
        else:
            plt.scatter(x_points, y_points)
            plt.axis('equal')
        plt.show()

def plot_object(model, currentAxis=None, plot_point=None):
    plot_point_i = np.matmul(model.kinematics_equilibrium_dict['Cbi'].transpose(), plot_point)
    from matplotlib.patches import Rectangle
    import matplotlib.animation as animation

    rectangle = Rectangle((plot_point_i[0],
                           plot_point_i[1]),
                          model.object.a,
                          model.object.b,
                          angle=model.state_equilibrium_dict['theta']*180/np.pi,
                          alpha=.35)
    rectangle.set_facecolor([0,0,1])
    rectangle.set_edgecolor([0,0,0])
    currentAxis.add_patch(rectangle)
    return currentAxis

def plot_friction_cone(model, contact_id, fc1, fc2, color='r', alpha=1):
    # 1. print friction cone
    fc1_local = np.matmul(model.kinematics_equilibrium_dict['C' + contact_id + 'i'].transpose(), fc1)
    fc2_local = np.matmul(model.kinematics_equilibrium_dict['C' + contact_id + 'i'].transpose(), fc2)
    arrow_fc1 = plt.arrow(model.kinematics_equilibrium_dict['r' + contact_id + 'i_i'][0] - fc1_local[0],
                         model.kinematics_equilibrium_dict['r' + contact_id + 'i_i'][1] - fc1_local[1],
                         fc1_local[0],
                         fc1_local[1],
                         width=0.000001,
                         length_includes_head=True,
                         alpha=alpha,
                         color=color)
    arrow_fc2 = plt.arrow(model.kinematics_equilibrium_dict['r' + contact_id + 'i_i'][0] - fc2_local[0],
                         model.kinematics_equilibrium_dict['r' + contact_id + 'i_i'][1] - fc2_local[1],
                         fc2_local[0],
                         fc2_local[1],
                         width=0.000001,
                         length_includes_head=True,
                         alpha=alpha,
                         color=color)
    return arrow_fc1, arrow_fc2

def plot_force_vector(model, contact_id, force_vec, color='b', alpha=1, n_initial=5):

    force_vec_normalized = force_vec / (n_initial * 75)
    force_vec_i = np.matmul(model.kinematics_equilibrium_dict['C' + contact_id + 'i'].transpose(), force_vec_normalized)
    force_vect_i = np.matmul(model.kinematics_equilibrium_dict['C' + contact_id + 'i'].transpose(),
                         np.array([force_vec_normalized[0], 0]))
    force_vecn_i = np.matmul(model.kinematics_equilibrium_dict['C' + contact_id + 'i'].transpose(),
                         np.array([0, force_vec_normalized[1]]))

    try:
        arrow = plt.arrow(model.kinematics_equilibrium_dict['r' + contact_id + 'i_i'][0] - force_vec_i[0],
                          model.kinematics_equilibrium_dict['r' + contact_id + 'i_i'][1] - force_vec_i[1],
                          force_vec_i[0],
                          force_vec_i[1],
                          width=0.0001,
                          length_includes_head=True,
                          alpha=alpha,
                          color=color)
        return arrow
    except:
        pass

def plot_force_decomposition(model, contact_id, force_vec, n_initial):
    force_vec_normalized = force_vec / (n_initial * 75)
    force_vec_i = np.matmul(model.kinematics_equilibrium_dict['C' + contact_id + 'i'].transpose(), force_vec_normalized)
    force_vect_i = np.matmul(model.kinematics_equilibrium_dict['C' + contact_id + 'i'].transpose(),
                         np.array([force_vec_normalized[0], 0]))
    force_vecn_i = np.matmul(model.kinematics_equilibrium_dict['C' + contact_id + 'i'].transpose(),
                         np.array([0, force_vec_normalized[1]]))

    arrow_x = plt.arrow(model.kinematics_equilibrium_dict['r' + contact_id + 'i_i'][0] - force_vecn_i[0],
                      model.kinematics_equilibrium_dict['r' + contact_id + 'i_i'][1] - force_vecn_i[1],
                      force_vecn_i[0],
                      force_vecn_i[1],
                      width=0.0001,
                      length_includes_head=True,
                      alpha=.5,
                      color=[0, 0, 1])
    try:
        arrow_y = plt.arrow(model.kinematics_equilibrium_dict['r' + contact_id + 'i_i'][0] - force_vect_i[0],
                          model.kinematics_equilibrium_dict['r' + contact_id + 'i_i'][1] - force_vect_i[1],
                          force_vect_i[0],
                          force_vect_i[1],
                          width=0.0001,
                          length_includes_head=True,
                          alpha=.5,
                          color=[1, 0, 0])
        return arrow_x, arrow_y
    except:
        return arrow_x, None

def plot_models(model_list, plot_point=None, filepath=None, color_list = ['g']*10, is_show=True):
    from matplotlib import pyplot as plt
    plt.figure()
    currentAxis = plt.gca()
    #1. plot object
    currentAxis = plot_object(model_list[0], currentAxis, plot_point=plot_point)
    model_color_list = ['b', 'r']
    alpha_list = [0.4, 1]
    n_initial = model_list[0].state_equilibrium_dict['n1']
    for model_counter, model in enumerate(model_list):
        for counter, name in enumerate(model.contact_dict['names']):
            nu = model.contact_dict['friction'][counter]
            force_vec = model.state_equilibrium_dict['f'+name]
            #2. plot ground forces
            scalar_mult = .03
            fc1 = scalar_mult * np.array([nu, 1])
            fc2 = scalar_mult * np.array([-nu, 1])
            color = color_list[counter]
            is_cone = True
            #2. plot friction cone
            if is_cone:
                arrow_fc1, arrow_fc2 = plot_friction_cone(model, name, fc1, fc2, color=model_color_list[model_counter], alpha=alpha_list[model_counter])
            #3. plot applied force
            reaction_force = plot_force_vector(model,
                                               name,
                                               force_vec,
                                               color=color,
                                               alpha=alpha_list[model_counter],
                                               n_initial=n_initial)

    reaction_force = plot_force_vector(model, 'i',
                                       model.FCM_i,
                                       color='r',
                                       alpha=alpha_list[model_counter],
                                       n_initial=n_initial)
    plt.axis('equal')
    plt.xlim((-.2, .2))
    plt.ylim((-.2, .2))

    if is_show:
        plt.show()
    if filepath is not None:
        plt.savefig(filepath)

class Object():
    def __init__(self, a, b, m):
        self.a = a
        self.b = b
        self.m = m

def initialize_pulling_tactile_setup():
    _object = Object(a=0.144,
                          b=0.09,
                          m=0.1)
    # 2. define contact configuration
    contact_dict = {}
    contact_dict['names'] = ['1', '2', '3']
    contact_dict['locations'] = [np.array([0, _object.b / 2]),
                                      np.array([-_object.a / 2, -_object.b / 2]),
                                      np.array([_object.a / 2, -_object.b / 2])]

    contact_dict['friction'] = [0.3, 0.2, 0.2]
    contact_dict['fc_angle'] = [np.pi, 0, 0]
    contact_dict['angle_frame'] = ['body', 'body', 'body']
    contact_dict['angle_control'] = [False, False, False]
    contact_dict['normal_control'] = [True, False, False]
    optimization_equilibrium_parameters = {'H': 10 * np.diag([1, 1, 1, 1, 1, 1, 1e-3, 1e-3, 1e-3]),
                                                    'q': -np.array([0, 0, 0, 0, 0, 0, 1., 1., 1.])
                                                    }

    optimization_control_parameters  = {'dn1': 2,
                                        'dtheta1':5 * np.pi / 180,
                                        'dtheta2':5 * np.pi / 180,
                                        'n1_limits':[0, 5],
                                        'theta1_limits':[0 * 180 / np.pi, 90 * 180 / np.pi],
                                        'theta2_limits':[0 * 180 / np.pi, 90 * 180 / np.pi],
                                        'alpha_weights':[50,100,50]
                                       }
    return _object, contact_dict, optimization_equilibrium_parameters, optimization_control_parameters

def initialize_levering_tactile_setup():
    _object = Object(a=0.09,
                          b=0.144,
                          m=0.1)
    # 2. define contact configuration
    contact_dict = {}
    contact_dict['names'] = ['1', '2', '3']
    contact_dict['locations'] = [np.array([-_object.a / 2, _object.b / 2]),
                                      np.array([_object.a / 2, _object.b / 2]),
                                      np.array([_object.a / 2, -_object.b / 2])]

    contact_dict['friction'] = [0.3, 0.3, 0.2]
    contact_dict['fc_angle'] = [-np.pi / 2, np.pi / 2, 0]
    contact_dict['angle_frame'] = ['body', 'body', 'world']
    contact_dict['angle_control'] = [True, True, False]
    contact_dict['normal_control'] = [True, False, False]
    optimization_equilibrium_parameters = {'H': 10 * np.diag([1, 1, 1, 1, 1, 1, 1e-3, 1e-3, 1e-3]),
                                                    'q': -np.array([0, 0, 0, 0, 0, 0, 1., 1., 1.])
                                                    }

    optimization_control_parameters  = {'dn1': 2,
                                        'dtheta1':5 * np.pi / 180,
                                        'dtheta2':5 * np.pi / 180,
                                        'n1_limits':[0, 5],
                                        'theta1_limits':[0 * 180 / np.pi, 90 * 180 / np.pi],
                                        'theta2_limits':[0 * 180 / np.pi, 90 * 180 / np.pi],
                                        'alpha_weights':[50,100,50]
                                       }
    return _object, contact_dict, optimization_equilibrium_parameters, optimization_control_parameters

def initialize_grasping_tactile_setup():
    _object = Object(a=0.144,
                          b=0.09,
                          m=0.1)
    # 2. define contact configuration
    contact_dict = {}
    contact_dict['names'] = ['1', '2']
    contact_dict['locations'] = [np.array([-_object.a / 2, 0]),
                                 np.array([_object.a / 2, 0]),]

    contact_dict['friction'] = [0.3, 0.3]
    contact_dict['fc_angle'] = [-np.pi/2, np.pi/2]
    contact_dict['angle_frame'] = ['body', 'body']
    contact_dict['angle_control'] = [False, False]
    contact_dict['normal_control'] = [True, False]
    optimization_equilibrium_parameters = {'H': 10 * np.diag([1, 1, 1, 1, 1e-3, 1e-3]),
                                                    'q': -np.array([0, 0, 0, 0, 1., 1.])
                                                    }

    optimization_control_parameters  = {'dn1': 2,
                                        'dtheta1':5 * np.pi / 180,
                                        'dtheta2':5 * np.pi / 180,
                                        'n1_limits':[0, 5],
                                        'theta1_limits':[0 * 180 / np.pi, 90 * 180 / np.pi],
                                        'theta2_limits':[0 * 180 / np.pi, 90 * 180 / np.pi],
                                        'alpha_weights':[50,100,50]
                                       }
    return _object, contact_dict, optimization_equilibrium_parameters, optimization_control_parameters

class TactileControl():
    def __init__(self, _object, contact_dict, optimization_equilibrium_parameters, optimization_control_parameters):
        #1. initialize model
        self._object = _object
        self.contact_dict = contact_dict
        self. optimization_equilibrium_parameters = optimization_equilibrium_parameters
        self.optimization_control_parameters = optimization_control_parameters
        self.model_initial = active_model(object=self._object, contact_dict=self.contact_dict)
        self.plot_point = np.array([-self._object.a / 2, -self._object.b / 2])

    def solve_equilibrium(self, control_input=[3, 0, 0], state_vec=[0,0,0], is_show=False):
        n1 = control_input[0]
        theta1 = control_input[1]
        theta2 = control_input[2]
        x = state_vec[0]
        y = state_vec[1]
        theta = state_vec[2]

        # 1. compute equilibrium
        self.model_equilibrium = copy.deepcopy(self.model_initial)
        self.model_equilibrium.set_nominal_configuration(equilibrium_dict={'x': x,
                                                                  'y': y,
                                                                  'theta': theta,
                                                                  'n1': n1,
                                                                  'theta1': theta1,
                                                                  'theta2': theta2, })
        equilibrium_dict = get_equilibrium_forces(self.model_equilibrium,
                                                  optimization_parameters=self.optimization_equilibrium_parameters,
                                                  )
        self.model_equilibrium.linearize_motion_equations(equilibrium_dict=equilibrium_dict)
        if is_show:
            plot_models([self.model_equilibrium], plot_point=self.plot_point,
                        is_show=is_show)

    def solve_controller(self, state_vec=[0,0,0], is_show=False):
        self.model_control = copy.deepcopy(self.model_equilibrium)
        new_equilibiurm_dict = tactile_controller(self.model_control, self.optimization_control_parameters)
        new_equilibiurm_dict['x'] = state_vec[0]
        new_equilibiurm_dict['y'] = state_vec[1]
        new_equilibiurm_dict['theta'] = state_vec[2]
        self.model_control.set_nominal_configuration(equilibrium_dict=new_equilibiurm_dict)
        self.model_control.linearize_motion_equations(equilibrium_dict=new_equilibiurm_dict)
        if is_show:
            plot_models([self.model_equilibrium,
                        self.model_control],
                        plot_point=self.plot_point,
                        is_show=is_show)
        return new_equilibiurm_dict

if __name__ == "__main__":
    _object, contact_dict, optimization_equilibrium_parameters, optimization_control_parameters = initialize_levering_tactile_setup()
    # _object, contact_dict, optimization_equilibrium_parameters, optimization_control_parameters = initialize_pulling_tactile_setup()
    # _object, contact_dict, optimization_equilibrium_parameters, optimization_control_parameters = initialize_grasping_tactile_setup()
    tactile_control = TactileControl(_object, contact_dict, optimization_equilibrium_parameters, optimization_control_parameters)
    # tactile_control.solve_equilibrium(control_input=[2, 0, 0 * np.pi / 180],
    #                                   state_vec=[0, 0, -0 * np.pi / 180],
    #                                   is_show=True)

    # tactile_control.solve_controller(state_vec=[0, 0, -0 * np.pi / 180],
    #                                  is_show=True)
    state = [0, 0, -16.3238 * np.pi / 180]
    tactile_control.solve_equilibrium(control_input=[2, 0.470581 * np.pi / 180, 16.4015 * np.pi / 180],
                                      state_vec=state,
                                      is_show=True)

    new_dict = tactile_control.solve_controller(state_vec=state,
                                     is_show=True)    
    for key in new_dict.keys():
        print(key, new_dict[key])


    # from IPython import embed
    # embed()

    #1. initialize model
    #2. online solve equilibrium and tactile controller


