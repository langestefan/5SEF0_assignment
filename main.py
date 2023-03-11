import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os

# import required .py files
import data_initialization
import minmax
import response
import constants as c

# set the logging level to INFO
logger = logging.getLogger(__name__)
logger.setLevel(c.LOG_LEVEL_MAIN)
logger.addHandler(c.handler)

# create a logging format
logging.basicConfig(
    format=("[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s"),
)



# INITIALIZE SCENARIO
# Length of simulation (96 ptu's per day and 7 days, 1 ptu = 15 minutes)
sim_length = 96 * 7 * 52
number_of_houses = 100


# (1) INITIALIZE DATA
# this creates the list of houses object and arranges all the earlier loaded data correctly
[list_of_houses, ren_share, temperature_data] = data_initialization.initialize(
    sim_length, number_of_houses
)

# array to store the total combined load of all households for each timestep
total_load = np.zeros(sim_length)


if __name__ == "__main__":
    logger.info("Starting simulation")
    t_start = time.time()

    for i in range(0, sim_length):
        # (2) determine the min and max power consumption of each DER during this timestep
        minmax.limit_ders(list_of_houses, i, temperature_data[i])

        for house in list_of_houses:
            # (3) now we determine the actual consumption of each DER
            # The PV wil always generate maximum power
            house.pv.consumption[i] = house.pv.minmax[1]

            # The EV will, if connected, always charge with maximum power
            house.ev.consumption[i] = house.ev.minmax[1]

            # The HP will keep the household temperature constant
            house.hp.consumption[i] = house.hp.minmax[0]

            house_load = (
                house.base_data[i]
                + house.pv.consumption[i]
                + house.ev.consumption[i]
                + house.hp.consumption[i]
            )
            # if the combined load is negative, charge the battery
            if house_load <= 0:
                house.batt.consumption[i] = min(-house_load, house.batt.minmax[1])
            else:
                # always immediately discharge the battery if the load is positive
                house.batt.consumption[i] = max(-house_load, house.batt.minmax[0])

        # (4) Response and update DERs for the determined power consumption
        total_load[i] = response.response(list_of_houses, i, temperature_data[i])

    logger.info(f"Finished simulation in {round(time.time() - t_start)} seconds")

reference_load = np.load("reference_load.npy")  # load the reference profile


def plot_grid(show: bool = False, path: str = c.sim_path):
    """
    Plot the total load and the daily power profile of the simulation and the reference profile

    :param show: If True, the plots will be shown. The plots will always be saved to the sim folder.
    """
    plt.title("Total Load Neighborhood")
    plt.plot(reference_load, label="Reference")
    plt.plot(total_load, label="Simulation")
    plt.xlabel("PTU [-]")
    plt.ylabel("Kilowatt [kW]")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, "total_load.png"), dpi=300)
    if show:
        plt.show()

    plt.figure()
    power_split = np.split(total_load, sim_length / 96)
    reference_split = np.split(reference_load, sim_length / 96)
    power_split = sum(power_split)
    reference_split = sum(reference_split)
    max_val = max(max(power_split), max(reference_split))
    power_split /= max_val
    reference_split /= max_val

    plt.title("Normalized Daily Power Profile")
    plt.plot(np.arange(1, 97) / 4, power_split, label="Simulation")
    plt.plot(np.arange(1, 97) / 4, reference_split, label="Reference")
    plt.xlabel("Hour [-]")
    plt.ylabel("Relative Power [-]")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, "daily_power_profile.png"), dpi=300)
    if show:
        plt.show()


def renewables():
    energy_export = abs(sum(total_load[total_load < 0] / 4))
    energy_import = sum(total_load[total_load > 0] / 4)
    renewable_import = sum(total_load[total_load > 0] * ren_share[total_load > 0]) / 4
    renewable_percentage = renewable_import / energy_import * 100

    # log the results
    logger.info(f"Energy Exported: {energy_export}")
    logger.info(f"Energy Imported: {energy_import}")
    logger.info(f"Renewable Share: {renewable_percentage}")


# display results
plot_grid()
renewables()
