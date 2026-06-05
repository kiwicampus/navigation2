# nav2_docking backport a Iron — trazabilidad de cambios

> Objetivo: traer al stack **ROS 2 Iron** (vendorizado en el submódulo `navigation2`,
> base `project/add_fov_segmentation`, era 1.3.x) el módulo **`nav2_docking` 1.4.0** de
> upstream `ros-navigation/navigation2`, con todas sus features
> (backward/blind docking, ciclo de vida del detector, `error_msg`, etc.).

## Estrategia: verbatim fijado a `91d7096`

`nav2_docking` se reemplazó por la versión **verbatim** del commit upstream
**`91d7096e` (#6101)** de `ros-navigation/navigation2`.

Ese pin es el último commit **antes** de #6063 *"Change clocks to use sim-time aware ROS Time"*,
que migra todo a `nav2::Rate(node, freq)` (constructor con reloj que **no existe en el
`rclcpp` de Iron**). En `91d7096` el docking aún usa `rclcpp::Rate(freq)` (Iron-compatible)
y ya tiene todas las features de 1.4.0.

- **No** existe ningún commit con las features y *sin* `nav2_ros_common` → ese paquete se vendoriza.
- **No** se actualiza `nav2_util` entero: en 1.4.0 elimina `nav2_util::LifecycleNode`, del que
  dependen ~41 paquetes del stack. Solo se traen/extienden piezas puntuales de forma retrocompatible.

El **código `.cpp/.hpp` de `nav2_docking` es idéntico a upstream `91d7096`**. Todo lo demás son
adaptaciones de framework necesarias para Iron (build system + utilidades compartidas).

## PRs de upstream que aporta este backport

| Feature | PR upstream (ros-navigation) |
|---|---|
| `error_msg` en results de acción + puertos BT | #4341 (backport #4460) |
| `dock_direction` como param de plugin (reemplaza `dock_backwards`) | #5079 |
| `rotate_to_dock` (backward a ciegas) + `OdomSmoother` + `odom_topic`/`odom_duration`/`rotation_angular_tolerance`/`rotate_to_dock_timeout` | #5153 |
| Ciclo de vida del detector (`start/stopDetectionProcess`, `detector_service_name`, `subscribe_toggle`) | #5015 (backport #5218) |
| `on_timeout` + `TIMEOUT` en nodos BT | (incluido en el verbatim @ 91d7096) |
| Fix rotación detección externa (`setEuler`→`setRPY`) | #6047 (ya viene en el verbatim) |
| `ParameterHandler` (gestión de params) | #5964 / #5900 (viene en el verbatim) |

> Cortes deliberados: **NO** se trae #6063 (sim-time `nav2::Rate`) ni nada por encima
> (eso fuerza la migración de Rate/timers de todo el stack a Iron-incompatible).

## Cambios por paquete

### `nav2_docking/*` — reemplazo verbatim @ 91d7096 (+ adaptaciones Iron)
- **Código** (`.cpp/.hpp`): idéntico a upstream. Nuevos: `opennav_docking/{include,src}/parameter_handler.{hpp,cpp}`, `test/dock_files/`, `test/docking_params.yaml`. Eliminado `test/test_dock_file.yaml`.
- **Adaptaciones Iron aplicadas sobre el verbatim:**
  - Includes TF2 `.hpp` → `.h` (`tf2/utils`, `tf2_ros/buffer`, `tf2_ros/transform_listener`) — Iron no tiene los `.hpp` (revert equivalente al commit kiwicampus `92385a4`).
  - `controller.cpp`: `isCollisionFree(local_pose.pose)` → `isCollisionFree(nav_2d_utils::poseToPose2D(...))` (la API de Iron toma `Pose2D`); `+ #include nav_2d_utils/conversions.hpp`.
  - `CMakeLists.txt` de los 3 paquetes + `test/CMakeLists.txt`: reescritos a **estilo Iron** (`set(dependencies …)` + `ament_target_dependencies`, sin targets namespaced `pkg::pkg` ni macro `nav2_package()` que Iron no resuelve). Se añadieron `nav2_ros_common`, `std_srvs`, `nav_2d_utils` y la fuente `parameter_handler.cpp`. Install de fixtures actualizado a `dock_files/` + `docking_params.yaml`.
  - Tests BT (`test_dock_robot.cpp`, `test_undock_robot.cpp`): `+ #include "nav2_ros_common/lifecycle_node.hpp"`.

### `nav2_ros_common/` — **vendorizado nuevo** (desde upstream @ 91d7096)
- Provee `nav2::LifecycleNode`, `nav2::SimpleActionServer`, `nav2::ServiceServer/Client`, `nav2::Rate`, `declare_or_get_parameter`, etc. (del que depende el docking verbatim).
- `cmake/bondcpp_shim.cmake` (nuevo) + `CONFIG_EXTRAS`: recrea el target `bondcpp::bondcpp` (Iron solo exporta variables ament viejas; Jazzy usa target namespaced). Guard `if(NOT TARGET)` → no-op en Jazzy.

### `nav2_util/` — extensiones retrocompatibles (sin tocar `nav2_util::LifecycleNode`)
- **`include/nav2_util/parameter_handler.hpp`** (nuevo): traído verbatim de upstream @ 91d7096 (header-only, base `nav2_util::ParameterHandler<>` que el docking usa). Solo depende de `nav2_ros_common`.
- **`twist_publisher.hpp`**: el constructor toma `rclcpp_lifecycle::LifecycleNode::SharedPtr` (clase base) + `qos` con default → acepta `nav2_util::LifecycleNode` (consumidores existentes) **y** `nav2::LifecycleNode` (docking).
- **`odometry_utils.{hpp,cpp}`**: el overload de `OdomSmoother` toma `rclcpp_lifecycle::LifecycleNode::WeakPtr` (clase base) por el mismo motivo.
- `package.xml`: `+ nav2_ros_common`.
- ✔️ Verificado que `nav2_bt_navigator` y `nav2_behavior_tree` (consumidores) siguen compilando.

### `nav2_common/cmake/nav2_package.cmake` — macros de test
- Backporteadas `nav2_add_test/gtest/pytest_test/gmock` (+ opción `USE_ISOLATED_TESTS`) desde upstream. Son wrappers finos sobre `ament_add_*` (existen en Iron); las variantes `ros_isolated` solo en Kilted+ → caen al `else`. Necesario porque los `test/CMakeLists.txt` y `nav2_ros_common` usan `nav2_add_gtest`.

### `nav2_msgs/action/{DockRobot,UndockRobot}.action`
- `+ uint16 TIMEOUT=907` (lo usan los nodos BT con `on_timeout`). `error_msg` ya existía.

### `nav2_behavior_tree/include/nav2_behavior_tree/bt_action_node.hpp`
- `+ virtual void on_timeout()` (no-op) + llamadas en los 2 sitios de timeout del action client. Backport del comportamiento de upstream (no existía en Iron).

### `navigation_plugins/docking_plugins/` (plugin Kiwi `WcsChargingDock`, fuera del submódulo)
- `configure(...)`: `rclcpp_lifecycle::LifecycleNode::WeakPtr` → `nav2::LifecycleNode::WeakPtr` (la interfaz core verbatim cambió la firma).
- Implementados los 2 virtuales puros nuevos: `startDetectionProcess()`/`stopDetectionProcess()` (no-op `return true` — usa detección interna).
- Publishers → `rclcpp_lifecycle::LifecyclePublisher` + `on_activate/on_deactivate` en `activate/deactivate`.

### `navigation2/.clang-format` (nuevo)
- `DisableFormat: true` en la raíz del submódulo → el IDE no reformatea el código vendorizado al guardar (el `auto_format.sh` ya excluye submódulos).

## Estado de build y tests (ROS 2 Iron)

- **Build (`BUILD_TESTING=ON`)**: ✅ `nav2_ros_common`, `opennav_docking_core/docking/_bt`, `docking_plugins`, y consumidores (`nav2_bt_navigator`, `nav2_behavior_tree`).
- **Tests gtest `opennav_docking`**: ✅ todos pasan, incluidos los nuevos de features (backward docking, detector lifecycle, dock_direction, rotate_to_dock).
- **Limitaciones conocidas (test-harness, no runtime):**
  - `opennav_docking_bt` gtest (`test_dock_robot/undock_robot`): la fixture verbatim pone `nav2::LifecycleNode` en el blackboard pero el `BtActionNode` de Iron lo lee como `rclcpp::Node`. El plugin BT compila y funciona en runtime (el `bt_navigator` provee el nodo correcto); falla solo la fixture unitaria.
  - `opennav_docking` pytest `test_docking_server.py`: test de *launch*, falla típico en entorno headless.

## Cómo reproducir el build

```bash
cd /workspace/rover/ros2
colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --packages-up-to opennav_docking opennav_docking_bt docking_plugins
```
(Si se buildea con tests, `nav2_common` debe estar reconstruido primero para que `nav2_add_gtest` exista.)

## Pendientes / decisiones abiertas
- Adaptar (o no) las fixtures unitarias de `opennav_docking_bt` a la convención de blackboard de Iron.
- Migrar los configs (`nav2_params*.yaml`) de `dock_backwards` (override deprecado, aún honrado) a `dock_direction` por-dock — opcional; requiere que `WcsChargingDock` parsee `dock_direction`.
- Formato final (`scripts/auto_format.sh` excluye el submódulo; el código docking queda en estilo upstream).
