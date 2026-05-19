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

    def create_property_layers(self):                               # create two property layers: (1) food (UF-HF) distribution layer (2) fee limitation layer 
                                                                                      
        x_gradient = np.linspace(0.1, 0.9, self.width)                                 # Left side starts UF-heavy, right side starts HF-heavy.
        food_dist_layer = np.tile(x_gradient, (self.height, 1))

        self.food_dist_layer = food_dist_layer

        self.grid.add_property_layer(
            PropertyLayer.from_data("food_dist_layer", self.food_dist_layer)
        )

        fee_limit_gradient = np.linspace(1.0, 0.0, self.width)                       # Fee-limit layer:
                                                                                     # Left side = poor / fee-limited.
                                                                                     # Middle = partial fee limitation.
                                                                                     # Right side = rich / little or no fee limitation.
        fee_limit_layer = np.tile(fee_limit_gradient, (self.height, 1))

        self.fee_limit_layer = fee_limit_layer

        self.grid.add_property_layer(
            PropertyLayer.from_data("fee_limit_layer", self.fee_limit_layer)
        )

    def create_agents(self):
        all_cells = list(self.grid.all_cells)
        self.random.shuffle(all_cells)

        if self.n_agents > len(all_cells):
            raise ValueError(
                "n_agents cannot be greater than the number of cells when capacity=1"
            )

        for i in range(self.n_agents):
            cell = all_cells[i]
            FoodAgent(self, cell)

    def step(self):                                                                   # one day of the simulation
        if self.steps == self.intervention_start_day:
            self.activate_intervention()

        if (
            self.slosh_on
            and self.steps > 0
            and self.steps % self.slosh_interval == 0
        ):
            self.apply_slosh()

        self.agents.shuffle_do("step")

        self.datacollector.collect(self)

    def activate_intervention(self):                                                # activate access intervention, norm intervention, or both
        if self.intervention_type in ["access", "both"]:
            self.access_intervention_on = True
            self.apply_access_intervention()

        if self.intervention_type in ["norm", "both"]:
            self.norm_intervention_on = True
            self.assign_norm_int_targets()

    def apply_access_intervention(self):                                            # apply one-time access intervention
        food_dist = self.grid.food_dist_layer.data

        low_HF_cells = food_dist < 0.5

        food_dist[low_HF_cells] = np.minimum(
            food_dist[low_HF_cells] + self.access_boost,
            1.0,
        )

        self.grid.food_dist_layer.data = food_dist

    def assign_norm_int_targets(self):                                             # select agents targeted by norm intervention
        for agent in self.agents:
            x, y = agent.cell.coordinate
            home_HF = self.grid.food_dist_layer.data[y, x]

            if self.norm_targeting == "national":
                if self.random.random() < self.norm_exposure_prob:
                    agent.selected_for_norm = True

            elif self.norm_targeting == "uf_heavy":
                if home_HF < 0.5 and self.random.random() < self.norm_exposure_prob:
                    agent.selected_for_norm = True

            else:
                raise ValueError("norm_targeting must be 'national' or 'uf_heavy'")

    def apply_slosh(self):
        food_dist = self.grid.food_dist_layer.data

        HF_heavy = food_dist >= 0.5
        UF_heavy = food_dist < 0.5

        food_dist[HF_heavy] = np.minimum(
            food_dist[HF_heavy] + self.slosh_effect,
            1.0,
        )

        food_dist[UF_heavy] = np.maximum(
            food_dist[UF_heavy] - self.slosh_effect,
            0.0,
        )

        self.grid.food_dist_layer.data = food_dist

    def get_accessible_HF(self, agent):
        x, y = agent.cell.coordinate

        values = []

        # This is fixed by neighborhood location, not current HF ratio.
        fee_limit_sensitivity = self.grid.fee_limit_layer.data[y, x]

        for dx in range(-self.delivery_radius, self.delivery_radius + 1):
            for dy in range(-self.delivery_radius, self.delivery_radius + 1):
                distance = abs(dx) + abs(dy)

                if distance > self.delivery_radius:
                    continue

                nx = x + dx
                ny = y + dy

                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue

                HF_value = self.grid.food_dist_layer.data[ny, nx]

                if distance > 0:
                    after_distance_penalty = 1 - (                                              # how much access remains after distance penalty 
                        fee_limit_sensitivity                                                   # poor (left side) vs. rich (right side) neighborhood
                        * self.delivery_fee_penalty_strength                                    # how much penalty strength matters
                        * (distance / self.delivery_radius)                                     # distance converted to 0~1 scale 
                    )

                    HF_value *= after_distance_penalty

                values.append(HF_value)

        return np.mean(values)

    def get_neighbor_agents(self, agent):
        neighbor_cells = self.grid.get_neighborhood(
            agent.cell.coordinate,
            radius=1,
            include_center=False,
        )

        neighbors = []

        for cell in neighbor_cells:
            neighbors.extend(cell.agents)

        return neighbors