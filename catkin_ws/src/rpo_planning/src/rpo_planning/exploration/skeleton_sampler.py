class SkeletonSampler:
    """Class to interface with neural network processes running that will
    take an environment observation and task specification as input
    and return a sequence of high-level skill actions to take as output.
    Uses LCM interprocesses communication
    """
    def __init__(self):
        self.skill_language = 
        self.publisher = 
        self.subscriber = 

    def predict(self, pointcloud, transformation_des):
        """Function to take the observation, in the form of a point cloud,
        along with the task specification, in the form of a desired
        rigid body transformation of the point cloud, and sent it to the 
        neural network process. Returns whatever skill sequence message
        that the neural network predicts

        Args:
            pointcloud (np.ndarray): [N x 3] array of [x, y, z] points
            transformation_des (np.ndarray): [4 x 4] transformation matrix

        Returns:
            list: Sequence of skill types (each element will be a string)
        """
        