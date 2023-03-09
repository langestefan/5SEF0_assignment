import numpy as np

heat_capacity_water = 4182  # [J/kg.K]


def limit_ev(house, i):
    # vehicle not home, so minmax = [0,0]
    if house.ev.session[i] == -1:
        house.ev.minmax = [0, 0]

    # vehicle is home, so determine min and max power
    else:
        # determine the ev charging session number
        session = int(house.ev.session[i])

        # minimum power required to charge the EV to the "required energy" in the time where the vehicle is home
        # always charge to 100% SoC
        required_energy = house.ev.size

        # multiply by four because of conversion from kWh to kW
        min_power = (max(0, (required_energy - house.ev.energy))) * 4

        # determine how many timesteps are left before the vehicle leaves
        time_left = house.ev.session_leave[session] - i

        # determine the min power by dividing the required power by the number of timesteps left
        min_power = min(house.ev.power_max, (min_power / time_left))

        # max charge power possible
        energy_left = house.ev.size - house.ev.energy

        # this max power is either what is left to fully charge the battery or the max charging capability
        max_power = min(house.ev.power_max, energy_left * 4)

        # store value in house
        house.ev.minmax = [min_power, max_power]


def limit_batt(house):
    # determine maximum discharge power either what is left or max discharge power
    # this value is negative, * 4 for conversion from energy to power
    dis_power = max(-(house.batt.energy * 4), -house.batt.power_max)

    # max charging power (either what is left to fully charge battery or max charge power), this value is positive
    charge_power = min((house.batt.size - house.batt.energy) * 4, house.batt.power_max)
    house.batt.minmax[0:2] = [dis_power, charge_power]


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
            limit_ev(house, i)

        # batt
        if house.ders[2] == 1:
            limit_batt(house)

        # hp
        if house.ders[3] == 1:
            limit_hp(house, i, T_ambient)
