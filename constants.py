import logging
import os
import time


# v2g = True:  EV can be charged and discharged
v2g = False
v2h = False

# off-peak ptu's of the day
# 0 = 00:00-00:15, ..., 95 = 23:45-00:00
wd_start = 0
wd_end = 15
off_peak_wd = [4 * wd_start, int(4 * wd_end)]  # @ workday
off_peak_we = [4 * 0, 4 * 24]  # @ weekend

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

# add formatter to handler
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s"
)
handler.setFormatter(formatter)

# range safety constant (% of battery size)
R_SAFETY = 0.5
P_MAX_CHARGE = 0.2
