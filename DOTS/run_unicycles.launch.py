import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource, FrontendLaunchDescriptionSource
import socket

#ros2 launch dots_example_controllers run_unicycles.launch.py name:=r13 neighbours_names:=r10,r14,r15,r12 

def generate_launch_description():

    robot_name_str      = socket.gethostname()
    robot_name_str      = robot_name_str.replace('-', '_')
    real_robot_name = robot_name_str

    declare_override_params = [
        DeclareLaunchArgument('M', default_value='0.1'),
        DeclareLaunchArgument('J', default_value='0.5'),
        DeclareLaunchArgument('K', default_value='20.0'),
        DeclareLaunchArgument('BETA', default_value='3.0'),
        DeclareLaunchArgument('INPUT', default_value='6.0'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('robot_name', default_value=real_robot_name), # real one
        DeclareLaunchArgument('name', default_value=real_robot_name), # r one
        DeclareLaunchArgument('neighbours_names', default_value=''),
        DeclareLaunchArgument('anchor', default_value='false')
    ]

    ld = LaunchDescription()

    for d in declare_override_params:
        ld.add_action(d)

    use_sim_time = LaunchConfiguration('use_sim_time')
    real_robot_name   = LaunchConfiguration('robot_name')
    name   = LaunchConfiguration('name')

    M     = LaunchConfiguration('M')
    J     = LaunchConfiguration('J')
    K     = LaunchConfiguration('K')
    BETA  = LaunchConfiguration('BETA')
    INPUT = LaunchConfiguration('INPUT')

    neighbours_names = LaunchConfiguration('neighbours_names')
    anchor = LaunchConfiguration('anchor')

    pkg_launch      = get_package_share_directory('dots_launch')

    setup_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_launch, 'launch', 'basic_vision.launch.py')), # fixed for vision
    )

    # Controller (launched on signal)
    launch_cmd = Node(
        package='dots_example_controllers',
        executable='launch_on_signal',
        namespace=robot_name_str,
        output='screen',
        parameters=[
            {'launch_package': 'dots_example_controllers'},
            {'launch_file': 'unicycles.launch.py'},
            {'anchor': anchor},
            {'name': name},
            {'neighbours_names': neighbours_names},
            {'M': M},
            {'J': J},
            {'K': K},
            {'BETA': BETA},
            {'INPUT': INPUT},
        ]
    )


    ld.add_action(setup_cmd)
    ld.add_action(launch_cmd)

    return ld
    

