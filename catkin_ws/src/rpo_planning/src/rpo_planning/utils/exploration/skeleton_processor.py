import random

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
        elif 'EOS' in skill:
            continue
        else:
            p_skill = skill
        processed_skeleton.append(p_skill)
    for p_skill in processed_skeleton:
        assert p_skill in available_skills, 'Obtained unavailable skill from prediction'
    return processed_skeleton
