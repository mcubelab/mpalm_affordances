from rpo_planning.utils.exploration.skeleton_processor import (
    process_skeleleton_prediction, separate_skills_and_surfaces)

class DiscreteAction:
    def __init__(self, skill_name, surface_name):
        """
        Constructor for DiscreteAction class, which contains
        unified information about the skill type and the name
        of the surface that is being used for placement

        Args:
            skill_name (str): Name of skill to represent, e.g. 'pull_right'
            surface_name (str): Name of placement surface to represent, e.g. 'table'
        """
        self.skill_name = skill_name
        self.surface_name = surface_name
        self.surface_pcd = None
        self._make_full_name()

    def _make_full_name(self):
        """
        Internal method to create full skill name out of individual skill and surface names
        """
        self.full_name = self.skill_name + '_' + self.surface_name

    def set_skill_name(self, skill_name):
        """
        Setter function for skill name

        Args:
            skill_name (str): Name of skill to represent, e.g. 'pull_right'
        """
        self.skill_name = skill_name
        self._make_full_name()

    def set_surface_name(self, surface_name):
        """
        Setter function for surface name

        Args:
            surface_name (str): Name of placement surface to represent, e.g. 'table'
        """
        self.surface_name = surface_name
        self._make_full_name()

    def set_surface_pcd(self, surface_pcd, surface_name=None):
        """
        Setter function for surface point cloud

        Args:
            surface_pcd (np.ndarray): Point cloud to associate with placement surface
                that is represented
            surface_name (str): Name of placement surface to represent. If None, uses
                the current placement surface name
        """
        self.surface_pcd = surface_pcd
        if surface_name is not None:
            self.set_surface_name(surface_name)
        self._make_full_name()


class SkillSurfaceSkeleton:
    """
    Class to represent all important aspects of a plan skeleton, including both
    skill types ('pull_right') and discrete placement surfaces ('table'). Attributes
    include various combinations of these along with their respective categorical
    indices that are used when representing them as input/output in a NN
    """
    def __init__(self, skeleton_full, skeleton_indices, skeleton_surface_pcds=None):
        self.skeleton_full_raw = skeleton_full
        self.skeleton_indices = skeleton_indices
        # want to represent the skill types separately from the placement surfaces
        self.skeleton_skills, self.skeleton_surfaces = separate_skills_and_surfaces(skeleton_full)
        if skeleton_surface_pcds is None:
            self.skeleton_surface_pcds = [None] * len(skeleton_full)
        else:   
            self.skeleton_surface_pcds = skeleton_surface_pcds 

        # from raw names, create internal skeleton with discrete actions
        self._setup_skeleton()

    def _setup_skeleton(self):
        """
        Function to process all raw information about skeleton into organized format using 
        DiscreteAction class
        """
        self.skeleton = []
        self.skeleton_full = []
        for i in range(len(self.skeleton_full_raw)):
            if self.skeleton_full_raw[i] == 'EOS':
                break
            action = DiscreteAction(
                self.skeleton_skills[i],
                self.skeleton_surfaces[i]
            ) 
            action.set_surface_pcd(self.skeleton_surface_pcds[i])
            self.skeleton.append(action)
            self.skeleton_full.append(self.skeleton_full_raw[i])