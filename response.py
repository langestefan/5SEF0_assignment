import numpy as np
import logging

import constants as c

logger = logging.getLogger(__name__)
logger.setLevel(c.LOG_LEVEL_MINMAX)
logger.addHandler(c.handler)

heat_capacity_water = 4182  # [J/kg.K]


def check_capacity(batt, name: str, i: int, house_id: int):
    """ Check if battery is too full or empty

    :param batt: battery object
    :param name: name of battery
    :param i: timestep
    :param house_id: house id
    """
    # double check if house battery is too full or empty
    if (0 > np.round(batt.energy, 4)) or (
        np.round(batt.energy, 4) > batt.size
    ):
        delta = round(batt.energy - batt.size, 3)
        if delta > 0:
            str = "full"
        else:
            str = "empty"
        logger.error(
            f"{name} batt too {str}: idx: {i} by amount: {delta} kWh for house: {house_id}"
        )



def response(list_of_houses: list, i: int, T_ambient: float):
    """
    This function is called every timestep and updates the DERs given the
    determined consumption.

    :param list_of_houses: list of houses
    :param i: timestep
    :param T_ambient: ambient temperature
    :return: Total load of all houses
    """
    total_load = 0

    for house in list_of_houses:
        # Base load and PV are already updated in the main
        # EV
        if house.ders[1] == 1:
            # skip first timestep because you will look back one timestep
            if i != 0:
                # if the vehicle left the house this timestep, substract the
                # energy lost during driving from the battery
                if house.ev.session[i] == -1 and house.ev.session[i - 1] != -1:
                    house.ev.energy -= house.ev.session_trip_energy[
                        int(house.ev.session[i - 1])
                    ]
                    if house.ev.energy <= 0:
                        logger.error(
                            f"EV energy below 0: {i} by amount: {round(house.ev.energy, 3)} kWh for house: {house.id}"
                        )
                        house.ev.energy = 0

            # save EV SoC for later analysis
            house.ev.energy_history[i] = house.ev.energy

            # update battery (note conversion from kW to kWh)
            house.ev.energy += house.ev.consumption[i] / 4

            # double check if battery is too full or empty
            check_capacity(house.ev, "EV", i, house.id)

        # HP
        if house.ders[3] == 1:
            heat_to_house_tank = (
                house.hp.consumption[i] * (1000 * 900)
            ) * house.hp.cop(house.hp.house_tank_T_set, T_ambient[0])

            # calculate the heat going from the house tank to the house
            dT_tank_house = (heat_to_house_tank - house.heat_demand_house[i]) / (
                house.hp.house_tank_mass * heat_capacity_water
            )
            house_tank_T = house.hp.house_tank_T + dT_tank_house

            # if demand is too great, the demand will be 0 but the tank will heat up
            if house_tank_T < house.hp.house_tank_T_min_limit:
                house.heat_demand_house[i] = 0

            # calculate the corresponding temperature in the house tank
            # converting the demand to the actual amount that goes in to the house
            heat_to_house = house.heat_demand_house[i]
            dT_tank_house = (heat_to_house_tank - heat_to_house) / (
                house.hp.house_tank_mass * heat_capacity_water
            )
            house.hp.house_tank_T = house.hp.house_tank_T + dT_tank_house

            heat_power_to_house = heat_to_house / 900

            # Update the house temperature given the heat power to house
            q_inter = heat_power_to_house * house.f_inter
            b = np.matmul(house.K_inv, q_inter) + house.b_part[i]
            house.temperatures = (
                np.matmul(house.super_matrix, house.temperatures - b)
                + house.a[i] * 900
                + b
            )

            house.hp.temperature_data[i] = np.array(
                [house.hp.house_tank_T, house.temperatures[1]]
            )

        # BATT
        if house.ders[2] == 1:

            # double check if battery is too full or empty
            check_capacity(house.batt, "House", i, house.id)

            # save batt SoC for later analysis
            house.batt.energy_history[i] = house.batt.energy
            # update battery (note conversion from kW to kWh)
            house.batt.energy += house.batt.consumption[i] / 4

        total_load += (
            house.base_data[i]
            + house.pv.consumption[i]
            + house.ev.consumption[i]
            + house.batt.consumption[i]
            + house.hp.consumption[i]
        )

    return total_load
