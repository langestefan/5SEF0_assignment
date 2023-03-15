import numpy as np
import matplotlib.pyplot as plt

#import required .py files
import data_initialization
import minmax
import response

#INITIALIZE SCENARIO
sim_length = 96*7*52 #Length of simulation (96 ptu's per day and 7 days)
number_of_houses = 100


#INITIALIZE DATA
[list_of_houses,ren_share,temperature_data] = data_initialization.initialize(sim_length, number_of_houses) #this creates the list of houses object and arranges all the earlier loaded data correctly
total_load = np.zeros(sim_length) #array to store the total combined load of all households for each timestep

hoog_tarief  = [0,1,2,3,4,5,6,17,18,19,20,21,22,23]
overgang = [7,8,9,10]

tarief = [5,5,4,4,3,3,4,3,2,1,1,1,1,1,2,3,4,4,5,5,5,5,5,5]

if __name__ == '__main__':

    print("start simulation")
    
    for i in range(0, sim_length):
        days = int(i / 96)
        ptu_in_day = i - (days*96)
        uur = int(ptu_in_day/4)
        
                

        minmax.limit_ders(list_of_houses, i, temperature_data[i]) # determine the min and max power consumption of each DER during this timestep

        for house in list_of_houses: #now we determine the actual consumption of each DER
            house.pv.consumption[i] = house.pv.minmax[1] #The PV wil always generate maximum power
            house.ev.consumption[i] = house.ev.minmax[0] + ((house.ev.minmax[1] - house.ev.minmax[0])*(1/tarief[uur]))
            house.hp.consumption[i] = house.hp.minmax[0] #The HP will keep the household temperature constant
            house_load = house.base_data[i] + house.pv.consumption[i] + house.ev.consumption[i] + house.hp.consumption[i]
            if house_load <= 0: #if the combined load is negative, charge the battery
                house.batt.consumption[i] = min(-house_load, house.batt.minmax[1])
            else: #always immediately discharge the battery
                if (tarief[uur] == 4):
                    house.batt.consumption[i] = max(-house_load, house.batt.minmax[0] *0.3)
                elif (tarief[uur] == 5):
                    house.batt.consumption[i] = max(-house_load, house.batt.minmax[0] *0.6)    
                elif (tarief[uur] in [1,2]):
                    house.batt.consumption[i] = house.batt.minmax[1] * 0.1
                else:
                    house.batt.consumption[i] = 0

        total_load[i] = response.response(list_of_houses, i, temperature_data[i]) #Response and update DERs for the determined power consumption
    print("finished simulation")

reference_load = np.load("reference_load.npy") #load the reference profile



def plot_grid():

    plt.title("Total Load Neighborhood")
    plt.plot(reference_load, label="Reference")
    plt.plot(total_load, label="Simulation")
    plt.xlabel('PTU [-]')
    plt.ylabel('Kilowatt [kW]')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    power_split = np.split(total_load, sim_length / 96)
    reference_split = np.split(reference_load, sim_length / 96)
    power_split = sum(power_split)
    reference_split = sum(reference_split)
    max_val = max(max(power_split),max(reference_split))
    power_split /= max_val
    reference_split /= max_val

    plt.title("Normalized Daily Power Profile")
    plt.plot(np.arange(1, 97) / 4, power_split, label = 'Simulation')
    plt.plot(np.arange(1, 97) / 4, reference_split, label = "Reference")
    plt.xlabel('Hour [-]')
    plt.ylabel('Relative Power [-]')
    plt.legend()
    plt.grid(True)
    plt.show()

def renewables():

    energy_export = abs(sum(total_load[total_load<0]/4))
    energy_import = sum(total_load[total_load>0]/4)
    renewable_import = sum(total_load[total_load > 0] * ren_share[total_load > 0])/4
    renewable_percentage = renewable_import/energy_import*100

    print("Energy Exported: ", energy_export)
    print("Energy Imported: ", energy_import)
    print("Renewable Share:", renewable_percentage)


plot_grid()

renewables()
