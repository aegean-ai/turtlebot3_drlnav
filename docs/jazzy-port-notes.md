# Porting Guide: ROS 2 Foxy + Gazebo 11 to ROS 2 Jazzy + Gazebo Harmonic

This document captures every API change required to port the turtlebot3_drlnav
stack from ROS 2 Foxy (with Gazebo Classic 11) to ROS 2 Jazzy (with Gazebo
Harmonic, i.e. `gz-sim`).

---

## 1. Launch Files (all `turtlebot3_drl_stage*.launch.py`)

### 1.1 Gazebo Classic packages no longer exist

| Foxy (Gazebo Classic)            | Jazzy (Gazebo Harmonic)                      |
|----------------------------------|----------------------------------------------|
| `gazebo_ros` package             | `ros_gz_sim` package                         |
| `gzserver.launch.py`            | `gz sim -r <world.sdf>` via `ExecuteProcess` |
| `gzclient.launch.py`           | GUI is part of `gz sim` (use `-r` flag)       |
| `gazebo_ros_pkgs` dependency     | `ros_gz`, `ros_gz_sim`, `ros_gz_bridge`      |

### 1.2 World file format change

- Gazebo Classic uses `.world` files (SDF wrapped in a `<sdf><world>` element
  but often stored as `.model`).
- Gazebo Harmonic uses plain `.sdf` files. World files should be converted to
  SDF 1.9+ format. Xacro can be used for parameterized worlds
  (see `turtlebot-maze` pattern: `.sdf.xacro` processed to temp `.sdf`).

### 1.3 Robot spawning

| Foxy                                        | Jazzy                                                |
|---------------------------------------------|------------------------------------------------------|
| Robot embedded in world `.model` file       | Robot spawned separately via `ros_gz_sim create`     |
| `gazebo_ros` spawn_entity plugin            | `ros_gz_sim` `create` executable with `-string` arg |

The Jazzy pattern (from `turtlebot-maze/turtlebot_spawner.launch.py`):
```python
Node(
    package="ros_gz_sim",
    executable="create",
    arguments=["-name", robot_name, "-string", Command([...xacro...]),
               "-x", x, "-y", y, "-z", z, "-R", roll, "-P", pitch, "-Y", yaw],
)
```

### 1.4 ros_gz_bridge required

Gazebo Harmonic topics are native `gz.msgs.*` types. A bridge node must be
launched to convert them to ROS 2 messages:

```python
Node(
    package="ros_gz_bridge",
    executable="parameter_bridge",
    parameters=[{
        "config_file": bridge_yaml,
        "expand_gz_topic_names": True,
        "use_sim_time": True,
    }],
)
```

A bridge YAML config is needed mapping topics:
```yaml
- topic_name: "/clock"
  ros_type_name: "rosgraph_msgs/msg/Clock"
  gz_type_name: "gz.msgs.Clock"
  direction: GZ_TO_ROS

- topic_name: "scan"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS

- topic_name: "odom"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS

- topic_name: "cmd_vel"
  ros_type_name: "geometry_msgs/msg/TwistStamped"
  gz_type_name: "gz.msgs.Twist"
  direction: ROS_TO_GZ
```

### 1.5 GZ_SIM_RESOURCE_PATH replaces GAZEBO_MODEL_PATH

```python
AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", models_dir)
```

### 1.6 cmd_vel type change

**CRITICAL**: In Gazebo Harmonic the DiffDrive plugin expects
`geometry_msgs/msg/TwistStamped` (not `Twist`). The bridge config maps
`TwistStamped` on ROS 2 side to `gz.msgs.Twist` on Gazebo side.

This means code publishing `Twist` must be changed to publish `TwistStamped`.
See section 3 below for the code change.

### 1.7 Pause/unpause physics

| Foxy                                   | Jazzy                                                    |
|----------------------------------------|----------------------------------------------------------|
| `std_srvs/srv/Empty` on `/pause_physics` and `/unpause_physics` | `ros_gz_interfaces/srv/ControlWorld` or use gz-transport directly |

In Gazebo Harmonic, physics pause/unpause is done via:
- `gz service -s /world/<world_name>/control --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --req 'pause: true'`
- Or via `ros_gz_interfaces/srv/ControlWorld` if the `ros_gz_bridge` exposes it.

### 1.8 New launch file template

Replace each `turtlebot3_drl_stageN.launch.py` with this pattern:

```python
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess,
    IncludeLaunchDescription, RegisterEventHandler, OpaqueFunction,
    AppendEnvironmentVariable,
)
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, FindExecutable
from launch_ros.actions import Node

STAGE = 4  # change per file

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    pkg_share = get_package_share_directory('turtlebot3_gazebo')

    # Write stage file
    with open('/tmp/drlnav_current_stage.txt', 'w') as f:
        f.write(str(STAGE))

    world_sdf = os.path.join(pkg_share, 'worlds',
                             f'turtlebot3_drl_stage{STAGE}', 'world.sdf')

    # If using xacro worlds:
    # world_sdf_tmp = tempfile.mktemp(prefix="drl_", suffix=".sdf")
    # world_xacro = ExecuteProcess(cmd=["xacro", "-o", world_sdf_tmp, world_sdf])

    # Set model paths
    set_gz_resource = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.path.join(pkg_share, "models")
    )

    # Launch Gazebo Harmonic
    gz_sim = ExecuteProcess(
        cmd=["gz", "sim", "-r", world_sdf],
        output="screen",
    )

    # Bridge
    bridge_config = os.path.join(pkg_share, 'configs', 'drl_bridge.yaml')
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            "config_file": bridge_config,
            "expand_gz_topic_names": True,
            "use_sim_time": True,
        }],
        output="screen",
    )

    # Spawn robot
    robot_sdf = os.path.join(pkg_share, 'urdf', 'turtlebot3_burger.sdf.xacro')
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "burger",
            "-string", Command([FindExecutable(name="xacro"), " ", robot_sdf]),
            "-x", "0", "-y", "0", "-z", "0.01",
        ],
        output="screen",
    )

    # Robot state publisher
    urdf_path = os.path.join(pkg_share, 'urdf', 'turtlebot3_burger.urdf')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "use_sim_time": use_sim_time,
            "robot_description": robot_description,
        }],
        output="screen",
    )

    return LaunchDescription([
        set_gz_resource,
        gz_sim,
        gz_bridge,
        spawn_robot,
        robot_state_pub,
    ])
```

---

## 2. `drl_gazebo.py` -- Gazebo service client changes

### 2.1 SpawnEntity / DeleteEntity services removed

In Gazebo Classic, `gazebo_msgs/srv/SpawnEntity` and `DeleteEntity` were
provided by `gazebo_ros`. These do **not exist** in Gazebo Harmonic.

**Replacement options:**

**Option A: Use `ros_gz_sim` create/remove executables from Python**
```python
import subprocess
# Spawn
subprocess.run(["ros2", "run", "ros_gz_sim", "create",
                "-name", name, "-file", sdf_path,
                "-x", str(x), "-y", str(y), "-z", str(z)])
# Remove
subprocess.run(["gz", "service", "-s",
                f"/world/{world_name}/remove",
                "--reqtype", "gz.msgs.Entity",
                "--reptype", "gz.msgs.Boolean",
                "--req", f'name: "{name}" type: MODEL'])
```

**Option B: Use gz-transport Python bindings**
```python
from gz.transport import Node as GzNode
from gz.msgs.entity_pb2 import Entity
from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.entity_factory_pb2 import EntityFactory

gz_node = GzNode()

# Spawn
req = EntityFactory()
req.sdf = sdf_string
req.name = "goal"
req.pose.position.x = goal_x
req.pose.position.y = goal_y
result, rep = gz_node.request("/world/default/create",
                              req, EntityFactory, Boolean, timeout=5000)

# Delete
req = Entity()
req.name = "goal"
req.type = Entity.MODEL
result, rep = gz_node.request("/world/default/remove",
                              req, Entity, Boolean, timeout=5000)
```

**Option C: Use ros_gz_interfaces (if available)**

Check if `ros_gz_interfaces` provides `SpawnEntity`/`DeleteEntity` services.
As of Jazzy, `ros_gz_interfaces` may expose a limited set. If not, use
Option A or B.

### 2.2 reset_simulation service removed

`/reset_simulation` from `gazebo_ros` does not exist in Gazebo Harmonic.

**Replacement**: Use gz-transport world control:
```python
# Reset world via CLI
subprocess.run(["gz", "service", "-s",
                f"/world/{world_name}/control",
                "--reqtype", "gz.msgs.WorldControl",
                "--reptype", "gz.msgs.Boolean",
                "--req", "reset: {all: true}"])
```

Or via Python gz-transport bindings:
```python
from gz.msgs.world_control_pb2 import WorldControl
req = WorldControl()
req.reset.all = True
gz_node.request("/world/default/control", req, WorldControl, Boolean, timeout=5000)
```

### 2.3 pause_physics service removed

See section 1.7 above. Replace `/pause_physics` calls with gz-transport
`WorldControl` with `pause: true`.

### 2.4 Python path in entity_dir_path

The hardcoded path fragment `python3.8/site-packages` must change to
`python3.12/site-packages` for Jazzy (Ubuntu 24.04 ships Python 3.12).
Better: use `ament_index_python.get_package_share_directory()` to find model
paths instead of string manipulation.

```python
# Old (fragile):
self.entity_dir_path = (...).replace(
    'turtlebot3_drl/lib/python3.8/site-packages/turtlebot3_drl/drl_gazebo',
    'turtlebot3_gazebo/share/turtlebot3_gazebo/models/...')

# New (robust):
from ament_index_python.packages import get_package_share_directory
self.entity_dir_path = os.path.join(
    get_package_share_directory('turtlebot3_gazebo'),
    'models', 'turtlebot3_drl_world', 'goal_box')
```

### 2.5 SDF model format

Gazebo Harmonic uses SDF 1.9+. Check that all `model.sdf` files in
`turtlebot3_gazebo/models/` use a compatible SDF version. The `<sdf
version="1.6">` tag should work, but features like `<sensor>` elements may
need schema updates.

---

## 3. `drl_environment.py` -- Environment node changes

### 3.1 cmd_vel: Twist to TwistStamped

In Gazebo Harmonic, the DiffDrive plugin expects `TwistStamped`. The
`ros_gz_bridge` maps `TwistStamped` (ROS 2) <-> `gz.msgs.Twist` (Gazebo).

```python
# Old:
from geometry_msgs.msg import Twist
twist = Twist()
twist.linear.x = action_linear
twist.angular.z = action_angular
self.cmd_vel_pub.publish(twist)

# New:
from geometry_msgs.msg import TwistStamped
twist = TwistStamped()
twist.header.stamp = self.get_clock().now().to_msg()
twist.header.frame_id = 'base_link'
twist.twist.linear.x = action_linear
twist.twist.angular.z = action_angular
self.cmd_vel_pub.publish(twist)
```

Update the publisher type accordingly:
```python
# Old:
self.cmd_vel_pub = self.create_publisher(Twist, self.velo_topic, qos)
# New:
self.cmd_vel_pub = self.create_publisher(TwistStamped, self.velo_topic, qos)
```

### 3.2 QoS profiles -- no changes required

`QoSProfile(depth=10)` and `qos_profile_sensor_data` remain unchanged
between Foxy and Jazzy. However, be aware that in Jazzy the default QoS for
some topics may differ. If subscriber/publisher QoS mismatches are observed,
explicitly set reliability/durability:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
```

### 3.3 rclpy API changes (Foxy to Jazzy)

| Item                               | Foxy                                    | Jazzy                                                              |
|------------------------------------|-----------------------------------------|--------------------------------------------------------------------|
| `rclpy.init(args=args)`           | Same                                    | Same (no change)                                                   |
| `rclpy.spin(node)`                | Same                                    | Same                                                               |
| `node.destroy()`                  | Works but deprecated                    | Use `node.destroy_node()`                                          |
| `numpy.core.numeric.Infinity`     | Works                                   | Use `float('inf')` or `math.inf` (numpy internal API)             |
| `call_async` + `spin_once` loop   | Common pattern                          | Still works; alternative: use `await` with async service calls     |
| `qos_profile_sensor_data`         | Imported from `rclpy.qos`              | Same import, same behavior                                         |

### 3.4 Clock topic

In Gazebo Classic, `/clock` is published by `gazebo_ros`. In Gazebo Harmonic,
`/clock` must be bridged explicitly via `ros_gz_bridge` config:
```yaml
- topic_name: "/clock"
  ros_type_name: "rosgraph_msgs/msg/Clock"
  gz_type_name: "gz.msgs.Clock"
  direction: GZ_TO_ROS
```
The `Clock` callback code itself does not change.

---

## 4. `drl_environment_real.py` -- Real robot environment

Minimal changes needed since this file does not interact with Gazebo:

- Change `node.destroy()` to `node.destroy_node()` (line 227 equivalent).
- No `cmd_vel` type change needed for real robot (real robot still uses `Twist`).
- The `from numpy.core.numeric import Infinity` used in the sim version is
  not present here, but the pattern should be avoided in new code.

---

## 5. `utilities.py` changes

### 5.1 SDF parsing for scan count

The `get_scan_count()` function parses the burger `model.sdf` to find lidar
sample count. In Gazebo Harmonic:

- The SDF schema may use `<lidar>` instead of `<ray>` for the sensor element.
- Check the new TurtleBot3 SDF for Harmonic. The path might be:
  `<sensor type="gpu_lidar">` with `<lidar><scan><horizontal><samples>`.

```python
# Old (Gazebo Classic):
link.find('sensor').find('ray').find('scan').find('horizontal').find('samples')

# New (Gazebo Harmonic):
sensor = link.find('sensor')
lidar = sensor.find('lidar') or sensor.find('ray')  # fallback
lidar.find('scan').find('horizontal').find('samples')
```

### 5.2 get_simulation_speed

The world file structure changes. Physics configuration in SDF 1.9+:
```xml
<physics name="default" type="dart">
  <real_time_factor>1</real_time_factor>
  <max_step_size>0.001</max_step_size>
</physics>
```
Parse accordingly (no `<world>` wrapper if reading the SDF directly).

### 5.3 Python path for model file

`os.getenv('DRLNAV_BASE_PATH')` based paths to model files still work if
the env var is set correctly. Consider using `get_package_share_directory()`
for robustness.

---

## 6. `package.xml` changes

### 6.1 `turtlebot3_drl/package.xml`

Add dependency on `zenoh` (for the new Zenoh bridge feature):
```xml
<!-- No ROS package for zenoh; it's a pip dependency -->
```

### 6.2 `turtlebot3_gazebo/package.xml`

Replace Gazebo Classic dependencies:
```xml
<!-- Old -->
<depend>gazebo_ros_pkgs</depend>

<!-- New -->
<depend>ros_gz</depend>
<depend>ros_gz_sim</depend>
<depend>ros_gz_bridge</depend>
<depend>ros_gz_interfaces</depend>
```

Remove or update the `<gazebo_ros>` export tag:
```xml
<!-- Old -->
<export>
  <gazebo_ros gazebo_model_path="${prefix}/models"/>
</export>

<!-- New: model path set via GZ_SIM_RESOURCE_PATH env var in launch -->
<export>
  <build_type>ament_cmake</build_type>
</export>
```

---

## 7. Dockerfile / build environment changes

| Item            | Foxy                            | Jazzy                                      |
|-----------------|---------------------------------|--------------------------------------------|
| Base OS         | Ubuntu 20.04                    | Ubuntu 24.04                               |
| Python          | 3.8                             | 3.12                                       |
| CUDA base       | `nvidia/cuda:11.3.1-base-ubuntu20.04` | `nvidia/cuda:12.x-base-ubuntu24.04` |
| ROS 2 install   | `ros-foxy-*`                    | `ros-jazzy-*`                              |
| Gazebo          | `ros-foxy-gazebo-ros-pkgs`      | `ros-jazzy-ros-gz`                         |
| PyTorch         | Pip install for CUDA 11.3       | Pip install for CUDA 12.x                  |

---

## 8. SDF / Model file migration

### 8.1 World files

Each `turtlebot3_drl_stageN/burger.model` must be converted to proper SDF:
- Rename to `.sdf` (or `.sdf.xacro` if parameterization is needed).
- Update `<sdf version="...">` to `1.9` or higher.
- Remove Gazebo Classic plugins (`<plugin name="gazebo_ros_..."`).
- Add Gazebo Harmonic system plugins in the world:
  ```xml
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
    <render_engine>ogre2</render_engine>
  </plugin>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  ```

### 8.2 Robot model

- The TurtleBot3 burger model SDF must use Gazebo Harmonic plugins:
  - `gz-sim-diff-drive-system` instead of `libgazebo_ros_diff_drive.so`
  - `gz-sim-joint-state-publisher-system` instead of `libgazebo_ros_joint_state_publisher.so`
  - `gz-sim-gpu-lidar-system` or `gz-sim-sensors-system` for LiDAR
- The `<sensor type="ray">` may need to become `<sensor type="gpu_lidar">`.

### 8.3 Obstacle / goal models

- The goal box `model.sdf` should work if it uses basic SDF geometry.
- Verify that `<visual>` and `<collision>` elements use SDF 1.9 compatible
  schema.

---

## 9. Summary of file-by-file changes

| File | Changes Required |
|------|-----------------|
| `turtlebot3_drl_stageN.launch.py` (x10) | Complete rewrite: `gz sim` + `ros_gz_bridge` + `ros_gz_sim create` |
| `drl_environment.py` | `Twist` -> `TwistStamped` for cmd_vel; `destroy()` -> `destroy_node()`; add Zenoh bridge |
| `drl_environment_real.py` | `destroy()` -> `destroy_node()` |
| `drl_gazebo.py` | Replace `SpawnEntity`/`DeleteEntity`/`reset_simulation` with gz-transport; fix Python path |
| `utilities.py` | Update SDF parsing for `<lidar>` tag; update world file parsing |
| `turtlebot3_gazebo/package.xml` | `gazebo_ros_pkgs` -> `ros_gz*` dependencies |
| `turtlebot3_drl/package.xml` | Add `ros_gz_interfaces` if using ROS service wrappers |
| `Dockerfile` | Ubuntu 24.04, Python 3.12, CUDA 12.x, `ros-jazzy-*` |
| World `.model` files | Convert to `.sdf`, add Harmonic system plugins |
| Robot `model.sdf` | Update plugins to `gz-sim-*-system` variants |
| Bridge config (NEW) | Create `drl_bridge.yaml` for clock, scan, odom, cmd_vel, obstacle/odom |

---

## 10. Migration order (recommended)

1. Convert world SDF files and robot model to Gazebo Harmonic format.
2. Create the `ros_gz_bridge` YAML config.
3. Rewrite one launch file (e.g., stage4) and verify Gazebo Harmonic launches.
4. Port `drl_gazebo.py` (spawn/delete/reset).
5. Port `drl_environment.py` (TwistStamped, destroy_node).
6. Port `utilities.py` (SDF parsing).
7. Update `package.xml` files.
8. Update Dockerfile.
9. Test end-to-end training loop.
10. Port remaining stage launch files.
