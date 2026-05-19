import numpy as np
import mesa

from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.property_layer import PropertyLayer

from agents import FoodAgent


def average_intake(model):                                                                  # average intake_score of entire population
    return np.mean([a.intake_score for a in model.agents])


def percent_healthy(model):                                                                 # share of healthy agents
    stats = [a.day_health_stat for a in model.agents if a.day_health_stat is not None]
    if not stats:
        return 0
    return stats.count("healthy") / len(stats)


def percent_unhealthy(model):                                                               # share of unhealthy agents
    stats = [a.day_health_stat for a in model.agents if a.day_health_stat is not None]
    if not stats:
        return 0
    return stats.count("unhealthy") / len(stats)


class FoodModel(mesa.Model):                                                                # world settings
    def __init__(
        self,
        width=50,
        height=50,
        n_agents=750,
        delivery_radius=3,                                                                
        delivery_fee_penalty_strength=0.5,                                                 # how strongly fee_penalty matters                                    
        intervention_start_day=30,                                                         # intervention starts mid-run 
        intervention_type="none",                                                          # "none", "access", "norm", "both"
        norm_targeting="national",                                                         # norm intervention will randomly target select eligible agents from 
                                                                                           # whole grid "national" or only uf_heavy areas
        norm_exposure_prob=0.3,                                                            # how many people are exposed to norm intervention 
        access_boost=0.25,                                                                 # access intervention boosts ideal_HO +0.25
        norm_boost=0.005,                                                                  # norm intervention boosts ideal_HO +0.005
        mimicry_effect=0.01,                                                               # mimicry increases or decreases ideal_HO by 0.01 
        own_outcome_strength=0.01,                                                         # previous day's choice increases or decreases ideal_HO by 0.01 
        access_weight=0.5,                                                                 # how much food choice depends on food access/environment
        orientation_weight=0.5,                                                            # how much food choice depends on personal health orientation
        healthy_threshold=6,                                                               # a day is considered "healthy" when intake_score = max 6
        slosh_on=False,                                                                    # False for main runs; True only for stress-test runs
        slosh_interval=30,                                                                 # Slosh happens every 30 days  
        slosh_effect=0.02,                                                                 # Slosh increases or decreases HF:UF ratio in cells by 0.02
        seed=None,                                                                         
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.n_agents = n_agents

        self.delivery_radius = delivery_radius
        self.delivery_fee_penalty_strength = delivery_fee_penalty_strength

        self.intervention_start_day = intervention_start_day
        self.intervention_type = intervention_type

        self.norm_targeting = norm_targeting
        self.norm_exposure_prob = norm_exposure_prob
        self.access_boost = access_boost
        self.norm_boost = norm_boost

        self.mimicry_effect = mimicry_effect
        self.own_outcome_strength = own_outcome_strength

        self.access_weight = access_weight
        self.orientation_weight = orientation_weight

        self.healthy_threshold = healthy_threshold

        self.slosh_on = slosh_on
        self.slosh_interval = slosh_interval
        self.slosh_effect = slosh_effect

        self.access_intervention_on = False
        self.norm_intervention_on = False

        self.grid = OrthogonalMooreGrid(
            (self.width, self.height),
            torus=False,
            capacity=1,
            random=self.random,
        )

        self.create_property_layers()
        self.create_agents()

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Average Intake Score": average_intake,
                "Percent Healthy": percent_healthy,
                "Percent Unhealthy": percent_unhealthy,
            }
        )

  
