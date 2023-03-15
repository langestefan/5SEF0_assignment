import numpy as np
import pickle
import logging

import constants as c

logger = logging.getLogger(__name__)
logger.setLevel(c.LOG_LEVEL_DATAINIT)
logger.addHandler(c.handler)


class House:
    def __init__(self, sim_length, id, baseload, pv_data, ev_data, hp_data):
        # General House Parameters
        self.id = id + 1  # give each household an ID
        self.ders = [1, 1, 1, 1]  # PV/EV/Batt/HP
        self.base_data = baseload  # load baseload data into house

        # Thermal Properties House
        self.temperatures = hp_data["temperatures"]
        self.T_setpoint = 293
        self.T_min = 18 + 273
        self.T_max = 21 + 273
        self.super_matrix = hp_data["super_matrix"][id]
        self.a = hp_data["alpha"][id]
        self.v_part = hp_data["v_part"][id]
        self.b_part = hp_data["b_part"][id]
        self.M = hp_data["M"][id]
        self.f_inter = hp_data["f_inter"]
        self.K_inv = hp_data["K_inv"][id]
        self.heat_demand_house = np.zeros(sim_length)

        # DERS
        self.pv = self.pv(pv_data, sim_length)
        self.ev = self.ev(ev_data, sim_length)
        self.batt = self.batt(sim_length)
        self.hp = self.hp(sim_length)

    class pv:
        def __init__(self, pv_data, sim_length):
            self.data = pv_data
            self.minmax = [0, 0]
            self.consumption = np.zeros(sim_length)

    class ev:
        def __init__(self, ev_data, sim_length):
            self.minmax = [0, 0]
            self.consumption = np.zeros(sim_length)

            self.power_max = ev_data["charge_cap"]  # kW
            self.size = ev_data["max_SoC"]  # kWh
            self.min_charge = ev_data["min_charge"]

            # energy in kWh in de battery, changes each timstep
            self.energy = ev_data["start_SoC"]
            # array to store previous battery state of charge for analyzing later
            self.energy_history = np.zeros(sim_length)
            # details of the location of the EV (-1 is not at home, other number indicates the session number)
            self.session = ev_data["EV_status"]
            # energy required during session
            self.session_trip_energy = ev_data["Trip_Energy"]
            # arrival times of session
            self.session_arrive = ev_data["T_arrival"]
            # leave times of session
            self.session_leave = ev_data["T_leave"]

            # v2g / v2h: for now use same setting for all EVs
            self.v2g = c.v2g
            self.v2h = c.v2h
            self.discharge = False

    class batt:
        def __init__(self, sim_length):
            # Based on Tesla Powerwall https://www.tesla.com/sites/default/files/pdfs/powerwall/Powerwall_2_AC_Datasheet_EN_NA.pdf
            self.minmax = [0, 0]
            self.consumption = np.zeros(sim_length)
            self.afrr = np.zeros(sim_length)

            self.power_max = 5  # kW
            self.size = 13.5  # kWh
            # energy in kWh in de battery at every moment in time
            self.energy = 6.25
            self.energy_history = np.zeros(sim_length)

    class hp:
        def __init__(self, sim_length):
            # building properties
            # Nominal capacity of heat pump installation
            self.nominal_power = 8000  # [W]
            # Minimal operational capacity for heat pump to run
            self.minimal_relative_load = 0.3  # [-]

            # house tank properties
            # Mass of buffer = Volume of buffer (Water)
            self.house_tank_mass = 120  # [kg]
            # Min temperature in the buffer tank
            self.house_tank_T_min_limit = 25  # [deg C]
            # Min temperature in the buffer tank
            self.house_tank_T_max_limit = 75  # [deg C]
            # Temperature setpoint in buffer tank
            self.house_tank_T_set = 40  # [deg C]
            # Initial temperature in buffer tank
            self.house_tank_T_init = 40  # [deg C]
            # Parameter initialized with initial temperature but changes over time
            self.house_tank_T = self.house_tank_T_init

            self.minmax = [0, 0]
            self.consumption = np.zeros(sim_length)
            self.temperature_data = np.zeros((sim_length, 2))
            self.actual = np.zeros(sim_length)

        def cop(self, T_tank, T_out):
            T_out = T_out - 273.15
            return (
                8.736555867367798
                - 0.18997851 * (T_tank - T_out)
                + 0.00125921 * (T_tank - T_out) ** 2
            )


def initialize(sim_length, number_of_houses):
    # Scenario Parameters
    np.random.seed(42)

    # Load pre-configured data
    f = open("data.pkl", "rb")
    scenario_data = pickle.load(f)
    baseloads = scenario_data["baseloaddata"]
    pv_data = scenario_data["irrdata"]
    ev_data = scenario_data["ev_data"]
    hp_data = scenario_data["hp_data"]
    temperature_data = hp_data["ambient_temp"]
    ren_share = scenario_data["ren_share"]

    # load day ahead prices
    f_dh = open("day_ahead_2020.pkl", "rb")
    day_ahead_prices = pickle.load(f_dh)["Day-ahead Price [EUR/MWh]"].values

    # determine distribution of data
    distribution = np.arange(number_of_houses)
    np.random.shuffle(distribution)

    # create a list containing all the household data and parameters
    list_of_houses = []
    for nmb in range(number_of_houses):
        list_of_houses.append(
            House(
                sim_length,
                nmb,
                baseloads[distribution[nmb]],
                pv_data[distribution[nmb]],
                ev_data[distribution[nmb]],
                hp_data,
            )
        )

    return [list_of_houses, ren_share, temperature_data, day_ahead_prices]
