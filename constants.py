import logging
import os
import time

# v2g = True:  EV can be charged and discharged
v2g = False
v2h = False

# if we use the home battery or not
USE_FLEX_HEATING = False
USE_HOME_BATTERY = False
USE_FLEX_BATT_CHARGING = False

# if we use real time consumption data or not
USE_REAL_CONS = False

# for plotting consumption data of a single house
PLOT_HOUSE = 1  # random.randint(1, 100)  # house number, starting at 1
PLOT_DAY = 355  # random.randint(0, 364)  # day of the year, starting at 0
PLOT_LEN = 3  # plot duration in days

# range safety constant (% of battery size)
R_SAFETY = 1.0

# how often we refresh the min/max scalar for the next 24 hours (in ptu's)
# for example: 48 means we refresh every 12 hours, 24 means we refresh every 6 hours... etc
PSCALER_PRICE_INT = 96

# PTU's of overlap between two consecutive price interval windows
PSCALER_PRICE_OVERLAP = 96
PSCALER_CONS_INT = 8
CONS_WINDOW = 48

# number of houses to simulate
N_HOUSES = 100

# LOGGING settings
LOG_LEVEL_GLOBAL = logging.INFO
LOG_LEVEL_MAIN = logging.INFO
LOG_LEVEL_MINMAX = logging.INFO
LOG_LEVEL_DATAINIT = logging.INFO

if LOG_LEVEL_MAIN > LOG_LEVEL_GLOBAL:
    LOG_LEVEL_MAIN = LOG_LEVEL_GLOBAL
if LOG_LEVEL_MINMAX > LOG_LEVEL_GLOBAL:
    LOG_LEVEL_MINMAX = LOG_LEVEL_GLOBAL
if LOG_LEVEL_DATAINIT > LOG_LEVEL_GLOBAL:
    LOG_LEVEL_DATAINIT = LOG_LEVEL_GLOBAL

current_time = time.strftime("%Y%m%d-%H%M%S")
sim_path = os.path.join("sims", current_time)
if not os.path.exists(sim_path):
    os.makedirs(sim_path)

log_location = os.path.join(sim_path, "sim" + ".log")
handler = logging.FileHandler(log_location)
handler.setLevel(logging.DEBUG)

# write entire constants.py to .txt file
with open("constants.py", "r") as f:
    constants = f.read()
with open(os.path.join(sim_path, "constants.txt"), "w") as f:
    f.write(constants)


# add formatter to handler
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s", datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
