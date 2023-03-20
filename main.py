import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import time
import os
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

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
    format=(
        "[%(asctime)s.%(msecs)03d] %(levelname)s [%(name)s:%(lineno)s] %(message)s"
    ),
    datefmt="%H:%M:%S",
)

# INITIALIZE SCENARIO
# Length of simulation (96 ptu's per day and 7 days, 1 ptu = 15 minutes)
sim_length = 96 * 7 * 52
number_of_houses = c.N_HOUSES


# (1) INITIALIZE DATA
# this creates the list of houses object and arranges all the earlier loaded data correctly
[
    list_of_houses,
    ren_share,
    temperature_data,
    day_ahead_prices,
] = data_initialization.initialize(sim_length, number_of_houses)

# array to store the total combined load of all households for each timestep
total_load = np.zeros(sim_length)


def off_peak_ptu(i: int):
    """
    Determine whether we are in a charging PTU or not

    :param i: timestep
    :return: boolean, True if we are in a charging PTU
    """
    # determine if we are in a workday or weekend
    ptu_in_day = i % 96
    weekday = (i // 96) % 7
    weekend = weekday > 4  # days 5 and 6 are weekend

    # determine if we are in a off-peak PTU depending on the day of the week/ptu in the day
    charge_session = c.off_peak_we if weekend else c.off_peak_wd
    return (ptu_in_day >= charge_session[0]) and (ptu_in_day < charge_session[1])


def ptu_to_hhmm(ptu: int):
    """
    Convert a ptu to a string in the format hh:mm

    :param ptu: ptu to convert
    :return: string in the format hh:mm
    """
    hour = ptu // 4
    minute = (ptu % 4) * 15
    return f"{hour:02d}:{minute:02d}"


def get_min_max_range(
    price_data: np.array, minmax_price_range: np.array, eps: float = 1e-6
):
    """
    Determine the min and max price range for the next 24 hours

    This is essentially normalizing the inverted price data to the range [0, 1]

    :param price_data: array with day-ahead prices
    :param minmax_price_range: array with current normalized price range
    :param eps: epsilon to avoid division by zero
    :return: normalized price data
    """
    # invert the price data
    price_data = -price_data
    min_price = np.min(price_data)
    max_price = np.max(price_data)

    # normalize the price data to the range [0, 1] and return
    norm = (price_data - min_price) / (max_price - min_price + eps)
    minmax_price_range[: len(norm)] = norm
    return minmax_price_range


def plot_loads(data: pd.DataFrame, title: str):
    """
    Plot consumption of each DER for a single house for 24 hours (96 ptu's)

    :param house: house to plot (index in list_of_houses)
    :param data: DataFrame with consumption data + normalized price data
    :param title: title of the plot
    """
    if len(data) != 96:
        logger.error(f"Cannot plot data with {data.size} rows, expected 96")
        return

    # plot the consumption of each DER price on twinx
    fig, ax1 = plt.subplots()
    ax1.plot(data["pv"], label="PV", color="green")
    ax1.plot(data["hp"], label="HP", color="red")
    ax1.plot(data["ev"], label="EV", color="blue")
    ax1.plot(data["batt"], label="Battery", color="purple")
    ax1.plot(data["appl"], label="Appl", color="grey")
    ax1.plot(data["house_total"], label="Total", color="black")
    ax1.set_xlabel("Time [HH:MM]")
    ax1.set_ylabel("Power [kW]")
    ax1.set_title(title)
    ax1.legend(loc="upper left")

    # bar plot for the price
    ax2 = ax1.twinx()
    ax2.bar(data.index, data["price"], label="Price", color="orange", alpha=0.3)
    ax2.set_ylabel("Norm. Charge Factor [0, 1]")
    ax2.legend(loc="upper right")

    # set the xticks to the correct time
    xticks = [ptu_to_hhmm(i) for i in range(0, 97, 12)]
    ax1.set_xticks(range(0, 97, 12))
    ax1.set_xticklabels(xticks)

    # yticks to integer values
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True)

    # save the plot
    # fig.tight_layout()
    save_path = os.path.join(c.sim_path, f"{title}.png")
    logger.info(f"Saving plot to {save_path}")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def get_opt_cons(der_obj: object, p_scalar: float):
    """
    Get the optimal consumption of a DER based on the normalized price

    :param der_obj: DER object
    :param p_scalar: normalized charge factor
    :return: min and max power
    """
    p_min = der_obj.minmax[0]
    p_max = der_obj.minmax[1]
    p_act = p_min + (p_max - p_min) * p_scalar
    return p_min, p_max, p_act


if __name__ == "__main__":
    logger.info("Starting simulation")
    t_start = time.time()

    # normalized prices to do congestion based charging
    ts = c.PTU_REFRESH_P_SCALAR_INT
    minmax_price_range = np.ones(96)

    # empty df for ploting consumption data
    plot_data = pd.DataFrame(
        columns=[
            "pv",
            "hp",
            "ev",
            "batt",
            "appl",
            "house_total",
            "price",
        ],
        index=range(96),
    )

    # run the simulation
    with logging_redirect_tqdm():
        for i in trange(0, sim_length):
            # (2) determine the min and max power consumption of each DER during this timestep
            minmax.limit_ders(list_of_houses, i, temperature_data[i])

            # get day-ahead price for this timestep
            day_ahead_price = day_ahead_prices[i]
            logger.debug(f"Day-ahead price: {day_ahead_price} (EUR/MWh)")

            # if we pass to the next day we need to determine the min and max price range for the next 24 hours
            if i % ts == 0:
                minmax_price_range = get_min_max_range(
                    day_ahead_prices[i : i + 96],
                    minmax_price_range,
                )
                logger.debug(
                    f"New min-max price range: \n{minmax_price_range} at: {ptu_to_hhmm(i%96)}"
                )

            # loop over all houses
            for house in list_of_houses:
                logger.debug(
                    f" --- House: {house.id} at timestep: {i} (off-peak: {off_peak_ptu(i)}, time[HH:MM]: {i%96*15//60:02}:{i%96*15%60:02}), weekday: {i//96%7} --- "
                )

                # (3) now we determine the actual consumption of each DER

                # EV
                v2h = house.ev.v2h
                v2g = house.ev.v2g

                # get the charging factor p_scaler for this timestep
                p_scaler = minmax_price_range[i % ts]

                # add a random peturbance to p_scaler
                logger.debug(f'Current p_scaler: {p_scaler}')

                # The PV wil always generate maximum power
                house.pv.consumption[i] = house.pv.minmax[1]

                # The HP has the options to heat more or keep the temperature constant
                if c.USE_FLEX_HEATING:
                    p_min, p_max, p_hp = get_opt_cons(house.hp, p_scaler)
                # The HP will keep the household temperature constant
                else:
                    p_hp = house.hp.minmax[0]

                house.hp.consumption[i] = p_hp

                house_base_load = (
                    house.base_data[i] + house.pv.consumption[i] + house.hp.consumption[i]
                )

                logger.debug(f"Base load house: {house_base_load:.2f} kW")

                # determine the EV consumption
                p_min, p_max, p_ev = get_opt_cons(house.ev, p_scaler**(1/3))

                # if we use v2h we can discharge the EV
                # if v2h:
                #     # we only discharge if there is a positive house load
                #     if house_base_load > 0:
                #         p_ev = max(p_ev, -house_base_load)
                #     else:
                #         p_ev = max(p_ev, 0)

                # house.ev.consumption[i] = p_ev
                house.ev.consumption[i] = house.ev.minmax[0]
                logger.debug(f"EV consumption: {house.ev.consumption[i]:.2f} kW")

                # add the EV consumption to the base load
                house_load = house_base_load + house.ev.consumption[i]

                # always charge the battery if we have a negative house load
                if c.USE_HOME_BATTERY:
                    if house_load <= 0:
                        p_solar = min(-house_load, house.batt.minmax[1])
                        logger.debug(
                            f"House {house.id} is charging the house battery with {house.batt.consumption[i]:.2f} kW"
                        )

                        # if p_scaler is high we can charge a little extra up to 1 kW
                        p_max = house.batt.minmax[1]
                        p_grid = min(p_max * (p_scaler**2), 1)
                        house.batt.consumption[i] = min(p_solar + p_grid, p_max)

                    # always discharge the battery
                    else:
                        house.batt.consumption[i] = max(-house_load, house.batt.minmax[0])
                        logger.debug(
                            f"House {house.id} is discharging the house battery with {house.batt.consumption[i]:.2f} kW"
                        )
                else:
                    house.batt.consumption[i] = 0
  
                # update house_load with the battery consumption
                house_load += house.batt.consumption[i]

                # make a plot of the consumption of each DER for house c.PLOT_HOUSE
                if house.id == c.PLOT_HOUSE and i // 96 == c.PLOT_DAY:
                    plot_data.loc[i % 96] = [
                        house.pv.consumption[i],
                        house.hp.consumption[i],
                        house.ev.consumption[i],
                        house.batt.consumption[i],
                        house.base_data[i],
                        house_load,
                        minmax_price_range[i % ts],
                    ]

            # (4) Response and update DERs for the determined power consumption
            total_load[i] = response.response(list_of_houses, i, temperature_data[i])

        logger.info(f"Finished simulation in {round(time.time() - t_start)} seconds")

        # plot the consumption of each DER for house c.PLOT_HOUSE
        plot_loads(plot_data, f"load_house{c.PLOT_HOUSE}_day{c.PLOT_DAY}")

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
