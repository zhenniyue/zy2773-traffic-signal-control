import numpy as np
import traci


class Grid:
    def __init__(self, grid_size=(8, 8), bounds=None):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        
        if bounds is None:
            self.bounds = (120, 120, 280, 280)
        else:
            self.bounds = bounds
            
        min_x, min_y, max_x, max_y = self.bounds
        self.width = max_x - min_x
        self.height = max_y - min_y
        self.cell_width = self.width / self.cols
        self.cell_height = self.height / self.rows
        
    def get_cell(self, x, y):
        min_x, min_y, max_x, max_y = self.bounds
        
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return None
            
        col = int((x - min_x) / self.cell_width)
        row = int((y - min_y) / self.cell_height)
        
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        
        return (row, col)
    
    def get_vehicle_grid(self):

        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            try:
                x, y = traci.vehicle.getPosition(veh_id)
                cell = self.get_cell(x, y)
                if cell is not None:
                    row, col = cell
                    grid[row, col] += 1
            except traci.exceptions.TraCIException:
                continue
                
        return grid
    
    def get_pedestrian_grid(self):

        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        
        person_ids = traci.person.getIDList()
        
        for person_id in person_ids:
            try:
                x, y = traci.person.getPosition(person_id)
                cell = self.get_cell(x, y)
                if cell is not None:
                    row, col = cell
                    grid[row, col] += 1
            except traci.exceptions.TraCIException:
                continue
                
        return grid
    
    def get_speed_grid(self):
        speed_grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        count_grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            try:
                x, y = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                cell = self.get_cell(x, y)
                if cell is not None:
                    row, col = cell
                    speed_grid[row, col] += speed
                    count_grid[row, col] += 1
            except traci.exceptions.TraCIException:
                continue
        
        # Calculate averages (avoid division by zero)
        mask = count_grid > 0
        speed_grid[mask] = speed_grid[mask] / count_grid[mask]
        
        return speed_grid
    
    def get_waiting_grid(self):
        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            try:
                x, y = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                
                if speed < 0.1:  # Vehicle is essentially stopped
                    cell = self.get_cell(x, y)
                    if cell is not None:
                        row, col = cell
                        grid[row, col] += 1
            except traci.exceptions.TraCIException:
                continue
                
        return grid
    
    def get_full_observation(self):
        return {
            'vehicles': self.get_vehicle_grid(),
            'pedestrians': self.get_pedestrian_grid(),
            'speeds': self.get_speed_grid(),
            'waiting': self.get_waiting_grid()
        }
    
    def get_stacked_observation(self):
        obs = self.get_full_observation()
        return np.stack([
            obs['vehicles'],
            obs['pedestrians'],
            obs['speeds'],
            obs['waiting']
        ], axis=0)
    
    def print_grid(self, grid, title="Grid"):
        print(f"\n{title}:")
        print("-" * (self.cols * 4 + 1))
        for row in grid:
            print("|" + "|".join(f"{val:3d}" for val in row) + "|")
        print("-" * (self.cols * 4 + 1))


if __name__ == "__main__":
    import os
    import sys
    
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SIM_DIR = os.path.join(SCRIPT_DIR, "simulation")
    SUMO_BIN = "/Applications/SUMO sumo-gui.app/Contents/MacOS/SUMO sumo-gui"

    SUMO_CMD = [
        SUMO_BIN,
        "-n", os.path.join(SIM_DIR, "network.net.xml"),
        "-r", os.path.join(SIM_DIR, "routes.rou.xml"),
        "-a", os.path.join(SIM_DIR, "traffic_light.add.xml"),   
        "--no-warnings", "true",
        "--start", 
]  
##   SUMO_CMD = ["/Applications/SUMO sumo-gui.app/Contents/MacOS/SUMO sumo-gui",
##             "-n", "simulation/network.net.xml",
##             "-r", "simulation/routes.rou.xml",
##             "--no-warnings", "true"]
    
    # Start SUMO
    traci.start(SUMO_CMD)
    
    # Create grid observer
    observer = Grid(grid_size=(8, 8), bounds=(120, 120, 280, 280))
    
    print("Running simulation with grid observation...")
    for step in range(500):
        traci.simulationStep()
        
        if step % 50 == 0:
            print(f"\n{'='*60}")
            print(f"Step {step} ({step} seconds)")
            print('='*60)
            
            obs = observer.get_full_observation()
            
            # Print vehicle counts
            observer.print_grid(obs['vehicles'], "Vehicle Counts")
            print(f"Total vehicles: {obs['vehicles'].sum()}")
            
            # Print waiting vehicles
            observer.print_grid(obs['waiting'], "Waiting Vehicles")
            print(f"Total waiting: {obs['waiting'].sum()}")
    
    traci.close()
    print("\nSimulation complete!")