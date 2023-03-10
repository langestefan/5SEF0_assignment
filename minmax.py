import numpy as np
import logging

import constants as c
from data_initialization import house

logger = logging.getLogger(__name__)
logger.setLevel(c.LOG_LEVEL_MINMAX)
logger.addHandler(c.handler)

heat_capacity_water = 4182  # [J/kg.K]


def determine_v2g_limits(i: int, house: house):
    """
    This function determines the minimum and maximum power limits when
    V2G is enabled and the EV is home.

    :param i: timestep
    :param house: house object
    :return: None
    """
    ptu_in_day = i % 96
    session = int(house.ev.session[i])
    time_left = house.ev.session_leave[session] - i

    # get current and the required energy for the EV to be charged for the next session
    max_energy = house.ev.size  # energy at 100% SOC
    required_energy = max(
        house.ev.session_trip_energy[session], c.R_SAFETY * max_energy
    )
    current_energy = house.ev.energy  # energy at current SOC
    energy_left = max_energy - house.ev.energy  # energy delta to 100% SOC
    delta_energy = required_energy - current_energy
    charging_ptu = (ptu_in_day > c.ev_charge_session[0]) and (
        ptu_in_day < c.ev_charge_session[1]
    )

    # check for errors
    if current_energy > max_energy:
        logger.error(f'Current energy > max energy: {current_energy} > {max_energy}')
    if required_energy > max_energy:
        logger.error(f'Required energy > max energy: {required_energy} > {max_energy}')
    if energy_left < 0:
        logger.error(f'Energy left < 0: {energy_left} < 0')

    logger.debug(
        f"Current energy: {round(current_energy)}, required energy: {round(required_energy)}, "
        f"delta energy: {round(delta_energy)}, charging_ptu: {charging_ptu}"
    )

    # charging ptu
    if charging_ptu:
        # delta energy is positive, we need to charge the EV
        if delta_energy > 0:
            logger.debug(f"Delta energy = {delta_energy}, charging EV")
            min_power = min(house.ev.power_max, (required_energy / time_left))
            max_power = min(house.ev.power_max, energy_left * 4)

        # delta energy is negative, we can discharge the EV
        # TODO: discharging is not required when there is no power deficit
        else:
            logger.debug(f"Delta energy = {delta_energy}, discharging EV")
            min_power = 0
            max_power = max(-house.ev.power_max, (delta_energy * 4 / time_left))

    # discharging ptu
    else:
        min_power = 0
        max_power = min(house.ev.power_max, energy_left * 4)

    return [min_power, max_power]

    # # we are in a ptu where the EV can be charged
    # if charging_ptu:
    #     # if the EV is sufficiently charged
    #     if current_energy >= required_energy:
    #         if current_energy >= c.R_SAFETY * max_energy:
    #             min_power = 0  # no discharge
    #             max_power = max(-house.ev.power_max, (delta_energy / time_left))
    #         else:
    #             # get to
    #             min_power = 0
    #     else:
    #         # min power by dividing the required power by the number of timesteps left
    #         # max power is either what is left to fully charge the battery or the max charging capability
    #         min_power = min(house.ev.power_max, (required_energy / time_left))
    #         max_power = min(house.ev.power_max, energy_left * 4)

    # # we are in a ptu where the EV can be discharged
    # else:
    #     # if the EV SOC is below the required SOC, we must charge the EV regardless of the PTU we are in
    #     # note that we do not expect (or like) this to happen at any point
    #     if current_energy <= required_energy:
    #         min_power = min(house.ev.power_max, (required_energy / time_left))
    #         max_power = min(house.ev.power_max, energy_left * 4)

    #     # if the EV SOC is above the required SOC, we can discharge the EV to the required SOC
    #     else:
    #         assert delta_energy < 0, "Delta energy should be negative"
    #         # we assume that max EV discharge power is equal to max EV charge power
    #         min_power = max(-house.ev.power_max, 0)
    #         max_power = max(-house.ev.power_max, (delta_energy / time_left))

    # return [min_power, max_power]


def limit_ev(house, i: int, v2g: bool):
    """
    This function limits the EV charging power based on the EV session and the
    battery state of charge.

    :param house: house object
    :param i: timestep
    :param v2g: boolean to determine if V2G is enabled
    :return: None
    """
    # vehicle not home, so minmax = [0,0]
    if house.ev.session[i] == -1:
        house.ev.minmax = [0, 0]

    # vehicle is home, so determine min and max power
    else:
        if v2g:
            min_power, max_power = determine_v2g_limits(i, house)
        else:
            # determine the ev charging session number
            session = int(house.ev.session[i])

            # always charge to 100% SoC
            required_energy = house.ev.size
            min_power = max(0, (required_energy - house.ev.energy)) * 4

            # determine how many timesteps are left before the vehicle leaves
            time_left = house.ev.session_leave[session] - i

            # determine the min power by dividing the required power by the number of timesteps left
            min_power = min(house.ev.power_max, (min_power / time_left))

            # max charge power possible
            energy_left = house.ev.size - house.ev.energy
            max_power = min(house.ev.power_max, energy_left * 4)

        # check if the min and max power are within the limits
        if (
            np.abs(min_power) > house.ev.power_max
            or np.abs(max_power) > house.ev.power_max
        ):
            logger.error(
                f"EV minmax power is outside the limits: {min_power}, {max_power}"
            )

        # store the min and max power in the house object
        # logger.debug(f"EV minmax: {min_power}, {max_power}")
        house.ev.minmax = [min_power, max_power]


def limit_batt(house):
    # determine maximum discharge power either what is left or max discharge power
    # this value is negative, * 4 for conversion from energy to power
    dis_power = max(-(house.batt.energy * 4), -house.batt.power_max)

    # max charging power (either what is left to fully charge battery or max charge power), this value is positive
    charge_power = min((house.batt.size - house.batt.energy) * 4, house.batt.power_max)
    house.batt.minmax = [dis_power, charge_power]


def limit_hp(house, i, T_ambient):
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
            limit_ev(house, i, c.v2g)

        # batt
        if house.ders[2] == 1:
            limit_batt(house)

        # hp
        if house.ders[3] == 1:
            limit_hp(house, i, T_ambient)
