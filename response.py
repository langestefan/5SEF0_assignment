import numpy as np

heat_capacity_water = 4182  # [J/kg.K]


def response(list_of_houses, i, T_ambient):
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
                        house.ev.energy = 0

            # save EV SoC for later analysis
            house.ev.energy_history[i] = house.ev.energy

            # update battery (note conversion from kW to kWh)
            house.ev.energy += house.ev.consumption[i] / 4

            # double check if battery is too full or empty
            if (0 > np.round(house.ev.energy, 4)) or (
                np.round(house.ev.energy, 4) > house.ev.size
            ):
                print("battery too empty/full: ", i)

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
