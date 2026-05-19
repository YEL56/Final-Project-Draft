import numpy as np

from mesa.discrete_space import CellAgent


# Initial agent traits
class FoodAgent(CellAgent):
    def __init__(self, model, cell):
        super().__init__(model, cell)

        self.ideal_HO = self.random.random()  #ideal HO: ideal Health Orientation (how much agents want to be healthy) is initially randomly assigned

        self.accessible_HF = 0                # HF: Healthy Food  #accessible_HF: access to healthy food (determined by 1. local HF ratio 2. HF ratio resulting 
                                              # from delivery radius expansion. This is composed of (1) radius that delivery app expands and (2) delivery fee 
                                              # limitation - which lessens the possibility of ordering from distant areas. Fee limitation affects poorest
                                              # neighborhoods more.) 
        self.HF_prob = 0                      # probability of choosing healthy food (HF)
        self.UF_prob = 0                      # probability of choosing unhealthy food (UF)

        self.intake_score = 0                 # Intake score is calculated as follows: HF = +1, UF = +2. If intake_score <= 6, healthy. Else: unhealhty
        self.day_health_stat = None           # whether the agent's food choices throughout the day (one round) was healthy vs unhealthy is based on intake_score
        self.previous_day_health_stat = None  # previous day health stat of neighbors determines micmiry. If majority of neighbors' health_state = healthy, 

        self.selected_for_norm = False          # This is the first of two interventions: norm intervention ticks up ideal_HO

# everything that happens in one day (round)
    def step(self):                                                       
        self.previous_day_health_stat = self.day_health_stat               # yesterday's health stat carries over to today's beginning health stat
        self.update_ideal_HO_from_neighbors()                              # based on neighbors' yesterday's health stat, update current ideal_HO

        self.accessible_HF = self.model.get_accessible_HF(self)            # accessible HF: calculated by looking at-local HF environment; 
                                                                           # envirnoment expanded by delivery (influenced by fee limitation whose importance is
                                                                           # influenced by distance and the neighborhood u live in (left side = more penalized 
                                                                           # bc it is the poorest neighbhorhood)). All adjusted HF values are then averaged into 
                                                                           # one effective HF access score 

        

        if self.selected_for_norm and self.model.norm_intervention_on:       # if agent is eligible for norm intervention AND norm intervention is turned on..
            self.ideal_HO += self.model.norm_push                            # self.ideal_HO ticks up by norm_push value (0.03)

        self.ideal_HO = np.clip(self.ideal_HO, 0, 1)                         # ideal_HO value is btw 0~1 

        self.HF_prob = (                                                     # probability of choosing HF = contribution from structural access + personal 
                                                                             # orientation (i assume both matter equally, hence both weights are 0.5)
            self.model.access_weight * self.accessible_HF
            + self.model.orientation_weight * self.ideal_HO
        )

        self.HF_prob = np.clip(self.HF_prob, 0, 1)                           # HF_prob value is btw 0~1
        self.UF_prob = 1 - self.HF_prob

        self.intake_score = 0                                                # intake_score = agents total score of HF & UF choices throughout the day. 
                                                                             # new day, new intake_score (hence 0) 

        for _ in range(3):                                                   # for three meals
            if self.random.random() < self.HF_prob:                          # if agent chooses HF for one meal = +1 
                self.intake_score += 1
            else:
                self.intake_score += 2                                       # if agent chooses UF for one meal = +2 

        self.intake_score += self.generate_snack_score()                     # include snacking in addition to three obligatory meals 

        if self.intake_score <= self.model.healthy_threshold:                # if intake_score = healthy threshold 
            self.day_health_stat = "healthy"                                 # agent had a HEALTHY DAY 
        else: 
            self.day_health_stat = "unhealthy"                               # if not, agent had UNHEALHTY DAY

        self.update_ideal_HO_from_own_outcome()                              # agent updates own ideal_HO from their day_health_stat (healthy eating lead to 
                                                                             # continued healthy eating, eating junk food leads to eating more junk food)
#======================================================================================================================
# influence from neighbors (mimicry)
    def update_ideal_HO_from_neighbors(self):                                # how to update ideal_HO from neighbors                       
        neighbors = self.model.get_neighbor_agents(self)                     # get agent's neighbors

        if not neighbors:
            return

        neighbors_previous_day_health_stat_list = [                          # make a list of previous day's health stat of neighbors                      
            n.previous_day_health_stat                                       # take each neighbor’s previous-day health result >>
            for n in neighbors                                               # do this for every neighbor  
            if n.previous_day_health_stat is not None                        # if previous day's health stat exists (on day 1 it may not exist, 
                                                                             # hence this line of code) 
        ]                                                                    # this will look something like [healhty, healthy, unhealthy...]

        if not neighbors_previous_day_health_stat_list:                      # for Day 1 when previous day's health stat does not exist
            return

        healthy_share = neighbors_previous_day_health_stat_list.count("healthy") / len(neighbors_previous_day_health_stat_list)  
                                                                            # count the share of healthy neighbors out of all neighbor's health stat list 


        if healthy_share > 0.5:                                              # if majority of neighbors are healthy eaters that day         
            self.ideal_HO += self.model.mimicry_effect                       # ideal_HO goes up by mimicry effect (0.01) 
        elif healthy_share < 0.5:                                            # if majority of neighbors are UNhealthy eaters that day
            self.ideal_HO -= self.model.mimicry_effect                       # ideal_HO goes down by mimicry effect (0.01) 
            
# influence from one's own previous day choice (addiction/habit) 
    def update_ideal_HO_from_own_outcome(self):                               # how to update ideal_HO from agent's own outcome                         
        if self.day_health_stat == "healthy":                                 # check if agent's own health status = healthy 
            self.ideal_HO += self.model.own_outcome_strength                  # self.ideal_HO increases by own outcome strength 
        elif self.day_health_stat == "unhealthy":                             # check if agent's own health status = unhealthy 
            self.ideal_HO -= self.model.own_outcome_strength                  # self.ideal_HO decreases by own outcome strength 

        self.ideal_HO = np.clip(self.ideal_HO, 0, 1)                          # force ideal_HO into valid bounds (btw 0~1) 
#snacking
    def generate_snack_score(self):                                            # snacking (three snack possibilities per day rolled into one "snack meal")
        if self.random.random() < self.HF_prob:
            return self.random.choice([0, 1, 2, 3])                            # no snack (0) one healhty snack (1) two healthy snakcs or one unhealhty snack (2) 
                                                                               # one healthy snack + one unhealthy snack (3) >>> these all count as relatively
                                                                               # healhty
        else:
            return self.random.choice([4, 5, 6])                               # the rest = unhealthy snacking