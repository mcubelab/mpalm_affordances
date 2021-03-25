import copy
import random

def separate_skills_and_surfaces(raw_skeletons_and_surfaces, skillset_cfg):
    """
    Function to process the prediction of a skeleton from the 
    neural net where each skill type contains information about
    both what skill should be executed and which of a discrete set
    of possible placement surfaces should be used as the target into a 
    separated list of just the skill types to execute along with the
    predicted placements

    Args:
        raw_skeletons_and_surfaces (list): Each element is string containing
            both the skill type and the name of the placement surface, connected
            by an underscore, e.g., 'pull_right_table'

    Returns:
        list: List of strings containing just skill names (e.g., 'pull_right')
        list: List of strings containing just surface names (e.g., 'table')
    """
    skill_names, surfaces = [], []
    for name in raw_skeletons_and_surfaces:
        if name in ['EOS', 'PAD', 'SOS']:
            continue
        full_name_split = copy.deepcopy(name).split('_')
        if full_name_split[-1] in skillset_cfg.SURFACE_NAMES:
            skill_name = ('_').join(full_name_split[:-1])
            surface_name = full_name_split[-1]  # we assume this is _table, so just get rid of the '_'
        else:
            skill_name = name
            surface_name = 'table' 
        skill_names.append(skill_name)
        surfaces.append(surface_name)
    return skill_names, surfaces


def process_skeleleton_prediction(raw_skeleton, available_skills):
    """
    Function to process the prediction of a skeleton from the neural
    net to be compatible with the expected skill names that are available.
    Deals with things like converting "pull" to "pull_right", and such.

    Args:
        raw_skeleton (list): Each element is a string, containing the name
            of what skill to use at that particular step
        available_skills (list): Each element is a string containing the name
            of a skill that is available to the robot

    Returns:
        list: List of strings that have been processed to match the available
            skills
    """
    processed_skeleton = []
    valid_pull_skills = [skill for skill in available_skills if 'pull' in skill]
    valid_push_skills = [skill for skill in available_skills if 'push' in skill]
    for i, skill in enumerate(raw_skeleton):
        if skill == 'pull' and 'pull' not in valid_pull_skills: 
            p_skill = random.sample(valid_pull_skills, 1)[0]
        elif skill == 'push' and 'push' not in valid_push_skills:
            p_skill = random.sample(valid_push_skills, 1)[0]
        elif skill in ['EOS', 'SOS', 'PAD']:
            continue
        else:
            p_skill = skill
        processed_skeleton.append(p_skill)
    for p_skill in processed_skeleton:
        assert p_skill in available_skills, 'Obtained unavailable skill from prediction'
    return processed_skeleton
