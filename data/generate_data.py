import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_robotics_data(n_samples=5000):
    robot_count = np.random.randint(2, 51, n_samples)
    task_complexity = np.random.uniform(1, 10, n_samples)
    communication_delay = np.random.uniform(0, 500, n_samples)
    swarm_density = np.random.uniform(0.1, 5.0, n_samples)
    obstacle_density = np.random.uniform(0, 30, n_samples)
    battery_level = np.random.uniform(10, 100, n_samples)
    signal_strength = np.random.uniform(-120, -30, n_samples)
    formation_type = np.random.choice(['Line', 'Wedge', 'Circle', 'Grid', 'Random'], n_samples)
    terrain_roughness = np.random.uniform(1, 10, n_samples)
    task_deadline = np.random.uniform(10, 300, n_samples)
    robot_speed = np.random.uniform(0.5, 5.0, n_samples)
    sensor_range = np.random.uniform(1, 50, n_samples)
    collision_probability = np.random.uniform(0, 0.5, n_samples)
    energy_efficiency = np.random.uniform(0.5, 1.5, n_samples)
    coordination_overhead = np.random.uniform(0, 40, n_samples)

    n_robots = robot_count / 50
    n_complexity = task_complexity / 10
    n_delay = 1 - communication_delay / 500
    n_density = swarm_density / 5
    n_obstacles = 1 - obstacle_density / 30
    n_battery = battery_level / 100
    n_signal = 1 - (signal_strength + 120) / 90
    n_terrain = 1 - terrain_roughness / 10
    n_deadline = task_deadline / 300
    n_speed = robot_speed / 5
    n_range = sensor_range / 50
    n_collision = 1 - collision_probability / 0.5
    n_efficiency = energy_efficiency / 1.5
    n_overhead = 1 - coordination_overhead / 40

    form_map = {'Line': 0, 'Wedge': 1, 'Circle': 2, 'Grid': 3, 'Random': 4}
    n_form = np.array([form_map[f] for f in formation_type]) / 4

    logit = (
        + 3.0 * n_robots
        + 2.5 * n_complexity
        + 3.5 * n_delay
        + 2.0 * n_density
        + 2.5 * n_obstacles
        + 3.0 * n_battery
        + 2.5 * n_signal
        + 1.5 * n_form
        + 2.0 * n_terrain
        + 1.5 * n_deadline
        + 2.0 * n_speed
        + 2.5 * n_range
        + 3.0 * n_collision
        + 2.0 * n_efficiency
        + 3.0 * n_overhead
        + 1.0 * n_delay * n_signal
        + 1.0 * n_robots * n_overhead
        + 0.5 * n_speed * n_range
        + 0.5 * n_collision * n_density
        - 20.0
    )

    prob_success = 1 / (1 + np.exp(-logit))
    mission_success = (prob_success > 0.5).astype(int)

    df = pd.DataFrame({
        'robot_count': robot_count,
        'task_complexity': task_complexity.round(1),
        'communication_delay': communication_delay.round(1),
        'swarm_density': swarm_density.round(2),
        'obstacle_density': obstacle_density.round(1),
        'battery_level': battery_level.round(1),
        'signal_strength': signal_strength.round(1),
        'formation_type': formation_type,
        'terrain_roughness': terrain_roughness.round(1),
        'task_deadline': task_deadline.round(1),
        'robot_speed': robot_speed.round(2),
        'sensor_range': sensor_range.round(1),
        'collision_probability': collision_probability.round(3),
        'energy_efficiency': energy_efficiency.round(2),
        'coordination_overhead': coordination_overhead.round(1),
        'mission_success': mission_success,
    })

    return df

df = generate_robotics_data(5000)

os.makedirs('data', exist_ok=True)
df.to_csv('data/robotics_coordination_data.csv', index=False)

print(f"Dataset shape: {df.shape}")
print(f"Mission success rate: {df['mission_success'].sum()} / {len(df)} ({df['mission_success'].mean()*100:.1f}%)")
print(f"Formation types: {df['formation_type'].value_counts().to_dict()}")
print(f"Sample data:\n{df.head()}")
print("\nMissing values:\n", df.isnull().sum())
