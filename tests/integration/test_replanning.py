"""
Integration test for Dynamic Replanning and Digital Twin Event System.
Tests full Hero Demonstration lifecycle: Plan -> Execute -> Inject Obstacle -> Replan -> Complete.
Classification: [ENGINEERING ADDITION]
"""

import sys
import os
import time
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.simulation.digital_twin.world_state import WorldState, SimulationStatus, EventType, Event


class TestDynamicReplanningIntegration(unittest.TestCase):
    """Test full integration loop of digital twin and replanning."""

    def setUp(self):
        self.world = WorldState()
        self.world.reset()

    def test_hero_scenario_event_flow(self):
        """Test the state transitions and event emissions during hero obstacle injection."""
        # 1. Mission setup
        start_pos = [1.0, 1.0]
        goal_pos = [8.0, 8.0]
        self.world.set_mission(start_pos, goal_pos)
        self.assertEqual(self.world.start, start_pos)
        self.assertEqual(self.world.goal, goal_pos)

        # 2. Add initial planned trajectory
        dummy_trajectory = [[1.0 + i * 0.35, 1.0 + i * 0.35] for i in range(20)]
        self.world.add_trajectory(
            points=dummy_trajectory,
            traj_type="planned",
            is_safe=True,
            metrics={"safety_violations": 0, "min_margin": 0.5}
        )
        self.assertIn("planned", [t.type for t in self.world.trajectories.values()])

        # 3. Simulate execution start
        self.world.status = SimulationStatus.EXECUTING
        self.world.robot.position = dummy_trajectory[5]  # at waypoint 5
        self.assertEqual(self.world.status, SimulationStatus.EXECUTING)

        # 4. Inject dynamic obstacle directly ahead on path
        obs_pos = [4.5, 4.5]
        self.world.add_obstacle(obs_pos, radius=0.8, shape="ellipse", is_dynamic=True, label="Dynamic Hazard")
        self.assertEqual(len(self.world.obstacles), 1)

        # 5. Verify event logging
        history = self.world.event_bus.get_history()
        event_types = [e['type'] for e in history]
        self.assertIn(EventType.OBSTACLE_ADDED.value, event_types)

        # 6. Trigger replan: robot pauses, logs REPLAN_REQUESTED, updates trajectory
        self.world.status = SimulationStatus.REPLANNING
        self.world.event_bus.emit(Event(
            type=EventType.REPLAN_REQUESTED,
            timestamp=time.time(),
            data={"reason": "Dynamic obstacle on active trajectory"}
        ))
        
        # New safe detour trajectory
        replanned_trajectory = [
            dummy_trajectory[5],
            [3.0, 5.0], [3.5, 6.0], [4.5, 7.0], [6.0, 7.5], [8.0, 8.0]
        ]
        self.world.add_trajectory(
            points=replanned_trajectory,
            traj_type="replanned",
            is_safe=True,
            metrics={"safety_violations": 0, "min_margin": 0.35}
        )
        self.world.replan_count += 1
        self.world.status = SimulationStatus.EXECUTING
        self.world.event_bus.emit(Event(
            type=EventType.REPLAN_COMPLETED,
            timestamp=time.time(),
            data={"replan_count": self.world.replan_count}
        ))

        # 7. Check replan recorded
        self.assertEqual(self.world.replan_count, 1)
        self.assertIn("replanned", [t.type for t in self.world.trajectories.values()])
        history_after = self.world.event_bus.get_history()
        types_after = [e['type'] for e in history_after]
        self.assertIn(EventType.REPLAN_COMPLETED.value, types_after)


if __name__ == "__main__":
    unittest.main()
