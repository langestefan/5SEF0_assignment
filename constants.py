import logging
import os
import time


# v2g = True:  EV can be charged and discharged
v2g = True

# ptu's of the day [0-95] when the EV can be charged
# 0 = 00:00-00:15, ..., 95 = 23:45-00:00
ev_charge_session = [4 * 5, 4 * 17]  # 5:00-17:00

# LOGGING settings
LOG_LEVEL_MAIN = logging.INFO
LOG_LEVEL_MINMAX = logging.INFO

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
R_SAFETY = 0.2
