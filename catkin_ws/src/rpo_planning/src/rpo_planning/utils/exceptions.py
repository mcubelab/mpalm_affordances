class RPOError(Exception):
    pass

class SkillApproachError(RPOError):
    def __init__(self, message='Failed to safely approach object'):
        self.message = message
        super(SkillApproachError, self).__init__(self.message)

class InverseKinematicsError(RPOError):
    def __init__(self, message='Failed to get valid IK solution'):
        self.message = message
        super(InverseKinematicsError, self).__init__(self.message)

class DualArmAlignmentError(RPOError):
    def __init__(self, message='Failed to align dual arm trajectory'):
        self.message = message
        super(DualArmAlignmentError, self).__init__(self.message)

class PlanWaypointsError(RPOError):
    def __init__(self, message='Failed to plan valid path through waypoints'):
        self.message = message
        super(PlanWaypointsError, self).__init__(self.message)

class MoveToJointTargetError(RPOError):
    def __init__(self, message='Failed to find feasible path to reach joint target'):
        self.message = message
        super(MoveToJointTargetError, self).__init__(self.message)
        
class PrimitivePlanningError(RPOError):
    def __init__(self, message='Failed to find low-level motion plan for manipulation primitive'):
        self.message = message
        super(PrimitivePlanningError, self).__init__(self.message)