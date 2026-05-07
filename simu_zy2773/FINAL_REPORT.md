# Final Simulation Report

## GitHub Repository
https://github.com/zhenniyue/zy2773-traffic-signal-control

---

# Updated / Final Simulation Files

## `routes_cosmos.rou.xml`
This is the main route file used for defining both pedestrian flows and vehicle flows in the simulation.

I manually added pedestrian movements across multiple directions of the intersection and designed custom traffic flows for surrounding roads. This file represents the core traffic scenario used in the final simulation.

---

## `randomTrips.rou.xml`
This file contains additional vehicle traffic generated using SUMO’s `randomTrips.py`.

Originally, vehicle flows were mainly added around the outer roads of the map, which caused the inner road network to have very little traffic activity. To improve traffic density and create a more realistic simulation environment, random trips were generated and added into the simulation.

---

## `cosmos-map_sidewalk2.net.xml`
This is the final updated network file containing the corrected sidewalk structure.

Additional sidewalks and pedestrian-accessible paths were added to support pedestrian routing and ensure that pedestrians could successfully traverse the intersection network. This became the final road network used in the project.

---

## `cosmos.sumocfg`
This is the main SUMO configuration file used to run the final simulation in `sumo-gui`.

It loads the final road network, route files, and simulation settings together into one executable simulation scenario.

---

# Final Simulation Setup

The final simulation combines:
- manually designed pedestrian flows,
- custom vehicle flows,
- randomly generated background traffic,
- and an updated sidewalk-enabled road network

to create a more realistic multi-agent traffic environment inside SUMO.