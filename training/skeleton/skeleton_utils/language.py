import sys

sys.path.append('..')
from skeleton_utils.skeleton_globals import SOS_token, EOS_token, PAD_token

class SkillLanguage:
    def __init__(self, name):
        self.name = name
        self.skill2index = {'PAD': PAD_token, 'SOS': SOS_token, 'EOS': EOS_token}
        self.skill2count = {}
        self.index2skill = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.n_skills = 3 
        
    def add_skill_seq(self, seq):
        for skill in seq.split(' '):
            self.add_skill(skill)
        
    def add_skill(self, skill, keep_count=False):
        if skill not in self.skill2index:
            self.skill2index[skill] = self.n_skills
            self.skill2count[skill] = 1
            self.index2skill[self.n_skills] = skill
            self.n_skills += 1
        else:
            if keep_count:
                self.skill2count[skill] += 1
            else:
                pass

    def process_data(self, data):
        for i in range(len(data)):
            outputs = data[i][1]
            for skill in outputs:
                self.add_skill(skill)
            