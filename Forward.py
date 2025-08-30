import numpy as np
from enum import IntEnum
import json


#Constants
ball_mass = .160 #kg
gf = np.array([0,0,-9.81])*ball_mass #N
ball_radius = 0.035 #m
cor = 0.5 #resititution
cf = 0.35 #friction

#initial characteristics
vmag_in = 22.22 #m/s
pos_in = np.array([0,0,2.0]) #m
phi_angle_in = 0
theta_angle_in = 3
spin_mag = 26#rps
spin_angle = -45 #deg

spin_mag_rads = spin_mag*2*np.pi
spin_angle_rad = spin_angle*np.pi/180
spin_vect = np.array([spin_mag_rads*np.sin(spin_angle_rad),spin_mag_rads*np.cos(spin_angle_rad),0])
theta_angle_rad = theta_angle_in*np.pi/180 #rad
phi_angle_rad = phi_angle_in*np.pi/180 #rad

class Comp(IntEnum):
    x = 0
    y = 1
    z = 2

v_in = np.array([vmag_in*np.cos(phi_angle_rad)*np.cos(theta_angle_rad),
                 vmag_in*np.cos(phi_angle_rad)*np.sin(theta_angle_rad),
                 vmag_in*np.sin(phi_angle_rad)])

def add_data(lst,sample,position,velocity):
    lst.append({
        "sample" : sample,
        "position":position,
        "velocity":velocity,
    })

def bounce_calc(v,spin_vect):
    vz_prev = v[Comp.z]
    v[Comp.z] = -v[Comp.z]*cor

    v_surf = np.cross([0,0,ball_radius],spin_vect)

    if np.linalg.norm(v_surf)>0:

        max_friction_impulse_magnitude = cf*ball_mass*(1+cor)*vz_prev

        impulse_stop_slip = np.linalg.norm(v_surf)*ball_mass
        friction_impulse_magnitude = min(max_friction_impulse_magnitude,impulse_stop_slip)
        
        impulse_unit_vector = -v_surf/np.linalg.norm(v_surf)
        friction_impulse = impulse_unit_vector*friction_impulse_magnitude

        v += friction_impulse/ball_mass

def air_sim(pos_in,v_in,sps,gf,mb):
    trajectory = []
    add_data(trajectory,0,pos_in.tolist(),v_in.tolist())
    curr_pos = pos_in
    curr_vel = v_in
    curr_acc = gf/mb
    sample = 1
    t_step = 1/sps
    
    while curr_pos[Comp.z] > ball_radius or curr_vel[Comp.z]>0:
        # t = sample/sps
        curr_vel += curr_acc*t_step
        curr_pos += curr_vel*t_step
        add_data(trajectory,sample,curr_pos.tolist(),curr_vel.tolist())
        sample+=1

    bounce_calc(curr_vel,spin_vect)

    while curr_pos[Comp.x] < 20.12:
        # t = sample/sps
        curr_vel += curr_acc*t_step
        curr_pos += curr_vel*t_step
        add_data(trajectory,sample,curr_pos.tolist(),curr_vel.tolist())
        sample+=1

    return trajectory



test_trajectory = air_sim(pos_in,v_in,60,gf,ball_mass)

output_file = r"Trajectory.json"
with open(output_file,"w") as file:
    json.dump(test_trajectory,file,indent=4) 
    



        

        
        

        

