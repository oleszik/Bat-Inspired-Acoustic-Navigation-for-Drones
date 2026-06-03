# Grid Frontier A* Integration Demo

This folder contains the first integration demo for the bat-inspired acoustic drone navigation ecosystem.

The demo connects four existing sibling repos:

- `drone-grid-simulator`: `GridWorld`, `Drone`, `SimpleRangeSensor`, `DiscoveredMap`, `RandomWorldGenerator`
- `drone-frontier-explorer`: `ExplorationGrid`, `FrontierDetector`, `FrontierScorer`
- `drone-path-planner`: `PlannerGrid`, `AStarPlanner`
- `drone-safety-monitor`: `SafetyMonitor`, `SafetyState`
- `drone-occupancy-mapper`: `OccupancyGrid`, `OccupancyMapper`, `MappingConfig`, `Pose2D`, `RangeReadings`

## What It Does

`grid_frontier_astar_demo.py` runs the original discovered-map exploration loop:

1. Sense with `SimpleRangeSensor`
2. Update the simulator `DiscoveredMap`
3. Convert the discovered map to an `ExplorationGrid`
4. Detect and cluster frontier cells
5. Score frontier clusters and choose a target
6. Convert the discovered map to a `PlannerGrid`
7. Plan a path with A*
8. Convert the next path step to a drone action
9. Build a `SafetyState` and call `SafetyMonitor.evaluate(state)`
10. Apply the safety decision before moving
11. Execute one safe action
12. Repeat until coverage, path, frontier, safety, or step limits stop the run

Short form:

```text
sense → map → frontier → plan → safety check → move
```

`grid_occupancy_frontier_astar_safety_demo.py` runs the newer occupancy-map exploration loop:

1. Sense with `SimpleRangeSensor`
2. Convert readings into `RangeReadings`
3. Convert drone state into `Pose2D`
4. Update `OccupancyMapper`
5. Classify `OccupancyGrid` cells as obstacle, free, or unknown
6. Convert the classified map to `ExplorationGrid`
7. Detect, cluster, and score frontiers
8. Convert the classified map to `PlannerGrid`
9. Plan with A*
10. Build `SafetyState` and call `SafetyMonitor.evaluate(state)`
11. Execute movement according to the safety decision

Short form:

```text
sense → probabilistic occupancy update → classified map → frontier → plan → safety → move
```

The difference is that the first demo uses `DiscoveredMap`, where cells become known directly from simulator rays. The occupancy demo uses probabilistic `OccupancyGrid` evidence first, then frontiers and A* consume the classified map.

## How To Run

From the umbrella repo root:

```bash
python integrations/grid_frontier_astar_demo.py
```

Run the occupancy-based demo:

```bash
python integrations/grid_occupancy_frontier_astar_safety_demo.py
```

Optional examples:

```bash
python integrations/grid_frontier_astar_demo.py --world-type doorway --max-steps 300
python integrations/grid_frontier_astar_demo.py --world-type cluttered_room --seed 11
python integrations/grid_frontier_astar_demo.py --allow-unknown-planning
```

The script automatically adds the sibling package directories to `sys.path` when they are present next to the umbrella repo.

## Expected Output

Each step prints:

- step number
- action
- drone position and heading
- discovered-map coverage
- frontier count
- planned path length
- latest range sensor readings
- safety action, severity, and reason

Final metrics are saved to:

```text
integrations/outputs/grid_frontier_astar_summary.json
integrations/outputs/safety_decision_log.json
integrations/outputs/grid_occupancy_frontier_astar_safety_summary.json
integrations/outputs/final_occupancy_map.json
integrations/outputs/safety_decision_log_occupancy_demo.json
```

The summary includes:

- `steps`
- `final_coverage`
- `collisions`
- `replans`
- `frontier_targets_chosen`
- `no_path_count`
- `safety_decisions_count`
- `safety_warning_count`
- `safety_critical_count`
- `termination_reason`
- `occupied_cells`, `free_cells`, and `unknown_cells` for the occupancy demo

## Safety

The demo only executes one immediate action from each planned path. Before every movement decision it builds a `SafetyState` from:

- drone position and direction
- front, left, and right range readings
- collision count
- no-path count
- steps without exploration progress
- coverage and mission step budget

`SafetyMonitor.evaluate(state)` can return:

- `CONTINUE`: execute the planned next action
- `ROTATE_SCAN`: rotate instead of moving forward
- `REPLAN`: skip movement and replan on the next loop
- `RETURN_HOME`: stop the demo with `safety_return_home`
- `STOP`: stop the demo with `safety_stop`
- `EMERGENCY_LAND`: stop the demo with `safety_emergency_land`

Before executing `FORWARD`, `choose_safe_next_action()` also checks the real `GridWorld` and refuses to move into a wall. If no safe path action is available, the drone rotates in place once.

## Known Limitations

- This is grid-only integration; it does not implement acoustic echo sensing.
- The original discovered-map demo is updated by a simple range sensor, not an occupancy mapper.
- The occupancy demo uses a simple range sensor as mapper input; it is not acoustic yet.
- Occupancy classification is probabilistic, so the occupancy demo defaults to a lower first-integration target coverage than the discovered-map demo.
- Unknown cells are blocked by default for A* planning.
- There is no RL policy, dashboard, matplotlib animation, or real drone control.
- Frontier scoring is intentionally simple and beginner-friendly.
- `RETURN_HOME` currently stops the demo; it does not yet plan a home path.

## Next Steps

- Add an occupancy mapper.
- Add a richer safety monitor policy and return-home planner.
- Replace the simple range sensor with an acoustic echo sensor.
- Add matplotlib or dashboard visualization.
- Add richer target selection and recovery behavior after the basic pipeline is stable.
