import numpy as np
from numpy import ceil, floor
import logging

import constants as c
from data_initialization import House

logger = logging.getLogger(__name__)
logger.setLevel(c.LOG_LEVEL_MINMAX)
logger.addHandler(c.handler)

heat_capacity_water = 4182  # [J/kg.K]


def determine_v2hg_limits(house: House, i: int):
    """
    This function determines the min and max power for V2H/G.

    In this case we can both charge and discharge the EV.

    The lower limit is determined by whether we need to charge to reach the required
    SOC or we are allowed to discharge to reach just above the required SOC.

    :param house: house object
    :param i: timestep
    :return: min_power, max_power
    """
    # energy left to charge battery to max = 100% SOC - current SOC
    max_energy = house.ev.size  # energy at 100% SOC
    current_energy = house.ev.energy  # energy at current SOC
    energy_to_max = max_energy - current_energy

    # current session and time left before leaving
    session = int(house.ev.session[i])
    time_left = house.ev.session_leave[session] - i

    # maximum charging power
    ev_power_max = house.ev.power_max

    # calculate energy quantities
    trip_energy = house.ev.session_trip_energy[session]
    safety_energy = c.R_SAFETY * max_energy
    required_energy = max(trip_energy, safety_energy)
    # TODO: consider setting required_energy = house.ev.size

    # delta energy is the energy we need to charge or discharge to reach the required SOC
    delta_energy = required_energy - current_energy

    logger.debug(
        f"EV [trip: {trip_energy:.2f}], [safety: {safety_energy:.2f}], [current: {current_energy:.2f}], "
        f"[required: {required_energy:.2f}], [delta: {delta_energy:.2f}], [to max: {energy_to_max:.2f}]"
    )

    # we must charge to reach the required SOC
    if delta_energy >= 0:
        logger.debug(f"EV delta energy = {delta_energy:.2f} >= 0, we must charge")

        # this max power is either what is left to fully charge the battery or the max charging capability
        max_power = min(ev_power_max, energy_to_max * 4)
        min_power = min(ev_power_max, (delta_energy * 4 / time_left))

    # we are allowed to discharge to reach just above the required SOC
    else:
        logger.debug(f"Delta energy = {delta_energy:.2f} < 0, we can discharge")
        # determine maximum discharge power based on required SOC
        # this value is negative, * 4 for conversion from energy to power
        dis_power = max(-ev_power_max, (delta_energy * 4 / time_left))

        # max charging power (either what is left to fully charge battery or max charge power)
        # TODO: what is best here? 
        charge_power = min(ev_power_max, (energy_to_max * 4 / time_left))

        # set min and max power
        min_power = dis_power
        max_power = charge_power

    logger.debug(f"EV [min, max] power: [{round(min_power, 2)}, {round(max_power, 2)}]")
    return min_power, max_power


def determine_normal_limits(house: House, i: int):
    """
    This function determines the min and max power for normal charging.

    :param house: house object
    :param i: timestep
    :return: min_power, max_power
    """
    # energy left to charge battery to max = 100% SOC - current SOC
    energy_to_max = house.ev.size - house.ev.energy

    # current session and time left before leaving
    session = int(house.ev.session[i])
    time_left = house.ev.session_leave[session] - i

    # determine min and max power
    required_energy = house.ev.size
    min_energy = max(0, (required_energy - house.ev.energy))
    min_power = min(house.ev.power_max, (min_energy * 4 / time_left))
    max_power = min(house.ev.power_max, energy_to_max * 4)
    return min_power, max_power


def limit_ev(house: House, i: int):
    """
    This function limits the EV charging power based on the EV session and the
    battery state of charge.

    :param house: house object
    :param i: timestep
    :return: [min_power, max_power]
    """
    # vehicle not home, so minmax = [0,0]
    if house.ev.session[i] == -1:
        logger.debug("EV not home")
        house.ev.minmax = [0, 0]

    # vehicle is home, so minmax = [min_power, max_power]
    else:
        v2h = house.ev.v2h
        v2g = house.ev.v2g

        # if v2h and v2g are not enabled we can't discharge the EV battery
        if not v2h and not v2g:
            min_power, max_power = determine_normal_limits(house, i)
        else:
            min_power, max_power = determine_v2hg_limits(house, i)

        house.ev.minmax = [min_power, max_power]


def limit_batt(house: House):
    # determine maximum discharge power either what is left or max discharge power
    # this value is negative, * 4 for conversion from energy to power
    dis_power = max(-(house.batt.energy * 4), -house.batt.power_max)

    # max charging power (either what is left to fully charge battery or max charge power), this value is positive
    charge_power = min((house.batt.size - house.batt.energy) * 4, house.batt.power_max)
    house.batt.minmax = [dis_power, charge_power]


def limit_hp(house: House, i, T_ambient):
    # calculate the heat demands for the house to keep temperature at setpoint
    v = np.matmul(house.super_matrix, house.temperatures) + house.v_part[i]
    heat_demand_house = max(
        0,
        (
            (house.T_setpoint - v[1])
            / (house.M[1, 1] * house.f_inter[1] + house.M[1, 2] * house.f_inter[2])
        )
        * 900,
    )

    # DETERMINING MIN -> keep the household and tank temperature constant
    tank_T_difference_no_hp = heat_demand_house / (
        house.hp.house_tank_mass * heat_capacity_water
    )
    tank_T_no_hp = house.hp.house_tank_T - tank_T_difference_no_hp

    # calculate the resulting heat to the tank
    # Provide no heat to house tank if its temperature is above the set temperature
    if tank_T_no_hp > house.hp.house_tank_T_set:
        min_heat_to_house_tank = 0
    else:
        min_dT_tank_house = house.hp.house_tank_T_set - tank_T_no_hp
        min_heat_to_house_tank = min(
            house.hp.nominal_power * 900,
            (house.hp.house_tank_mass * heat_capacity_water) * min_dT_tank_house,
        )

    # DETERIMINING HEAT_TANK -> keep temperature household temperature constant
    # but heat up tank temperature
    max_dT_tank_house = house.hp.house_tank_T_max_limit - tank_T_no_hp
    max_heat_to_house_tank = min(
        house.hp.nominal_power * 900,
        (house.hp.house_tank_mass * heat_capacity_water) * max_dT_tank_house
        + heat_demand_house,
    )

    min_power = min_heat_to_house_tank / house.hp.cop(
        house.hp.house_tank_T_set, T_ambient[0]
    )
    max_power = max_heat_to_house_tank / house.hp.cop(
        house.hp.house_tank_T_set, T_ambient[0]
    )
    minmax = np.array([min_power, max_power]) / (1000 * 900)

    # value needs to be stored for response.py
    house.heat_demand_house[i] = heat_demand_house
    house.hp.minmax = minmax


def limit_ders(list_of_houses, i, T_ambient):
    for house in list_of_houses:
        # pv
        if house.ders[0] == 1:
            # the minimum is always 0 and the maximum is what the generation data gives
            house.pv.minmax = [0, house.pv.data[i]]

        # ev
        if house.ders[1] == 1:
            limit_ev(house, i)

        # batt
        if house.ders[2] == 1:
            limit_batt(house)

        # hp
        if house.ders[3] == 1:
            limit_hp(house, i, T_ambient)
