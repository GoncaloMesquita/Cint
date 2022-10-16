import numpy as np
import skfuzzy.control as ctrl
import skfuzzy as fuzz
import matplotlib.pyplot as plt

core_temp = ctrl.Antecedent(np.arange(0, 101, 0.5),'core temperature')
clock_speed = ctrl.Antecedent(np.arange(0, 4.5, 0.5),'clock speed frequency')
fan_speed = ctrl.Consequent(np.arange(0,6001,1),'fan speed')



core_temp['cold'] = fuzz.trimf(core_temp.universe,[0, 0, 50])
core_temp['warm'] = fuzz.trimf(core_temp.universe,[30, 50, 70])
core_temp['hot'] = fuzz.trimf(core_temp.universe,[50, 100, 100])

clock_speed['low'] = fuzz.trimf(clock_speed.universe,[0, 0, 1.5])
clock_speed['normal'] = fuzz.trimf(clock_speed.universe,[0.5, 2, 3.5])
clock_speed['turbo'] = fuzz.trimf(clock_speed.universe,[2.5, 4, 4])

fan_speed['slow'] = fuzz.trimf(fan_speed.universe,[0, 0, 3500])
fan_speed['fast'] = fuzz.trimf(fan_speed.universe,[2500, 6000, 6000])


core_temp.view()
clock_speed.view()
fan_speed.view()
plt.show()

rule1 = ctrl.Rule(core_temp['cold'] & clock_speed['low'], fan_speed['slow'])
rule2 = ctrl.Rule(core_temp['cold'] & clock_speed['normal'], fan_speed['slow'])
rule3 = ctrl.Rule(core_temp['cold'] & clock_speed['turbo'], fan_speed['fast'])
rule4 = ctrl.Rule(core_temp['warm'] & clock_speed['low'], fan_speed['slow'])
rule5 = ctrl.Rule(core_temp['warm'] & clock_speed['normal'], fan_speed['slow'])
rule6 = ctrl.Rule(core_temp['warm'] & clock_speed['turbo'], fan_speed['fast'])
rule7 = ctrl.Rule(core_temp['hot'] & clock_speed['low'], fan_speed['fast'])
rule8 = ctrl.Rule(core_temp['hot'] & clock_speed['normal'], fan_speed['fast'])
rule9 = ctrl.Rule(core_temp['hot'] & clock_speed['turbo'], fan_speed['fast'])

# rule1.view()
# plt.show()

fan_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9])

speed_of_fan = ctrl.ControlSystemSimulation(fan_ctrl)

speed_of_fan.input['core temperature'] = 88.3
speed_of_fan.input['clock speed frequency'] = 1.75

speed_of_fan.compute()

print(speed_of_fan.output['fan speed'])
fan_speed.view(sim = speed_of_fan)
plt.show()
