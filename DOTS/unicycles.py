#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import random, math, numpy as np
from sensor_msgs.msg import PointCloud2
from rclpy.qos import qos_profile_sensor_data
import struct
from .shared_functions import load_robot_name_map
from std_msgs.msg import Float64MultiArray

# Global constants
#J = 0.5
#BETA = 0.3
ZETA = 0.05
#M = 0.1
DT = 0.005
MAX_SPEED = 0.3
MAX_ANG = 1.0

MUTE = False

# robot launch command:
# ros2 launch dots_example_controllers unicycles.launch.py robot_name:=r13 anchor:=false neighbor_topics:=r14  use_sim_time:=true M:=2.283 J:=0.896 K:=0.569 BETA:=2.744 INPUT:=132.0 

class DotNode(Node):
    def __init__(self, name='dot', placeholder_names=None, anchor=False):
        print('initialising node')
        super().__init__(name)

        name_map = load_robot_name_map()
        neighbours_names = []
        for logical in placeholder_names:
            if logical not in name_map:
                # raise RuntimeError(f"Unknown robot logical name: {logical}")
                neighbours_names.append(logical)
                self.get_logger().info(f"Unknown robot logical name: {logical}")
            else:
                neighbours_names.append(name_map[logical])

        self.get_logger().info(f"Real robot names: {neighbours_names}")

            # Determine neighbor topics - use odometry filtered instead?
        neighbours_topics = [f'/{n}/pos' for n in neighbours_names if n != name]
        print(f"[DEBUG] Neighbor topics for {name}: {neighbours_topics}")

       # Declare parameters
        self.declare_parameter('M')
        self.declare_parameter('J')
        self.declare_parameter('K')
        self.declare_parameter('BETA')
        self.declare_parameter('INPUT')

        # Load parameters with fallback defaults
        self.M     = float(self.get_parameter('M').value) if self.get_parameter('M').value != "__notset__" else 0.1
        self.J     = float(self.get_parameter('J').value) if self.get_parameter('J').value != "__notset__" else 0.5
        self.K     = float(self.get_parameter('K').value) if self.get_parameter('K').value != "__notset__" else 10.0 # going to be multiplied by a random float
        self.BETA  = float(self.get_parameter('BETA').value) if self.get_parameter('BETA').value != "__notset__" else 3.0
        self.INPUT = float(self.get_parameter('INPUT').value) if self.get_parameter('INPUT').value != "__notset__" else 6.0

        if not MUTE:
            self.get_logger().info(f"Parameters M: {self.M} J: {self.J} K: {self.K} BETA: {self.BETA} INPUT: {self.INPUT}")

        self.state = {'x': 0.0, 'y': 0.0, 's': 0.0, 'theta': 0.0, 'w': 0.0}
        #self.T_s = random.randint(50, 60) # 140 -> 180
        self.u = 0.0
        self.anchor = anchor
        self.name = name
        self.collision_range    = 0.75

        self.neighbours_positions = {}
        self.neighbours_rest_spring = {}
        self.collision_force = {'x': 0.0, 'y': 0.0}
        self.neighbour_randoms = {}

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub = self.create_publisher(Point, 'pos', 10)

        self.timer = self.create_timer(DT, self.update)

        # Global input
        self.global_sub = self.create_subscription(Float64MultiArray, '/global_input', self.global_u_callback, 10)
        #self.get_logger().info("Subscribed to /global_input")

        # Subscribe to the laser time of flight sensors topic. 
        # This must be set to the qos sensor profile since the data is sent as best effort, and the 
        # default qos is reliable, which produces no data traffic
        self.irtof_sub          = self.create_subscription(PointCloud2, 'sensor/scan', self.scan_callback, 
                                    qos_profile=qos_profile_sensor_data)

        # Odometry subscription
        #self.sub = self.create_subscription(Odometry, 'odometry/filtered', self.odom_callback, 10)
        self.sub = self.create_subscription(Odometry, 'odometry/filtered', self.odom_callback, 10)

        self.initial_pos_stored = False
        self.initial_pos = {'x': 0.0, 'y': 0.0}

        # Neighbors
        if neighbours_topics:
            # Print for debugging
            #print(f"[DEBUG] neighbor_topics before processing: {neighbor_topics}")
            # or using ROS2 logger
            self.get_logger().info(f"[DEBUG] Neighbor topics: {neighbours_topics}")

            if isinstance(neighbours_topics, str):
                neighbours_topics = [neighbours_topics]

            for topic in neighbours_topics:
                self.create_subscription(Point, topic, self.neighbour_callback(topic), 10)
                self.assign_neighbour_random(topic)
                self.get_logger().info(f"[DEBUG] Subscribed to neighbour topic: {topic}")  

        # assign spring to own position:
        self.assign_neighbour_random(self.name)

        now = self.get_clock().now()
        self.get_logger().info(f"ROS time: {now.to_msg().sec}.{now.to_msg().nanosec}")
        
    def scan_callback(self, msg):
        #self.get_logger().info(f"[DEBUG] Scan callback called")       
        # This function gets called with every new message on the laser scan topic.
        # Extract the data and work out if there is a collision, i.e range is less
        # than some amount, if so, select a new direction.

        # There are 16 sensors, make somewhere to store data from each one. The Gazebo
        # version of the laser scanner doesn't send data for points with no return detected
        # but the hardware does, so fill in the maximum range for each entry to start with.

        min_dist    = 0.1 #100.0
        min_vec     = np.array([0.0, 0.0])
        for i in range(msg.width):
            # Points within the pointcloud are actually locations in 3D space in the scanner
            # frame. Each is a float32 taking 12 bytes. 
            # Extract point, we don't care about the z coordinate
            [x, y, _]   = struct.unpack('fff', bytes(msg.data[i * msg.point_step : i * msg.point_step + 12]))

            # Keep track of the minimum distance, and save that vector
            vec     = np.array((x, y))
            dist    = np.linalg.norm(vec) 
            if dist < min_dist:
                min_vec     = vec # vector x y of object
                min_dist    = dist # pythagrous magnitude
       
        #self.get_logger().info('Min dist %8.3f vec [% 8.3f % 8.3f]' % (min_dist, min_vec[0], min_vec[1]))

        if min_dist < self.collision_range:
            #self.get_logger().info(f"[DEBUG] Within collision range")       
            # If too close to something, get the unit vector pointing away from the 
            # shortest irtof vector and use that to set the velocity, with a bit of random
            # angular velocity added
            unit_vector = -min_vec / min_dist # nx, nz
            penatration = (self.collision_range - min_dist)
            k_repulsion = 50.0 # tunable param
            Fx, Fy = k_repulsion * penatration * unit_vector
            self.collision_force['x'] = Fx
            self.collision_force['y'] = Fy
        
        else:
            self.collision_force['x'] = 0
            self.collision_force['y'] = 0


    def global_u_callback(self, msg: Float64MultiArray):
        self.u = msg.data[1]

    def assign_neighbour_random(self, topic_name): # create spring stiffness!
        rand_value = np.random.rand() * self.K
        self.neighbour_randoms[topic_name] = rand_value
        #self.get_logger().info(f"[DEBUG] Assigned random={rand_value:.3f} to neighbor {topic_name}")

    def neighbour_callback(self, topic_name):
        def callback(msg):
            if self.initial_pos_stored: # only begin storing neighbour information when your initial position is stored to allow for spring measurements
                if not (topic_name in self.neighbours_positions): # only once
                    self.neighbours_rest_spring[topic_name] = np.sqrt((msg.x - self.state['x']) ** 2 + (msg.y - self.state['y']) **2)
                    if not MUTE:
                        self.get_logger().info(f"Neighbour resting spring length set at: {topic_name} length: {self.neighbours_rest_spring[topic_name]}")
                        self.get_logger().info(f"My initial position: {self.initial_pos}")

                self.neighbours_positions[topic_name] = (msg.x, msg.y)
        return callback

    def odom_callback(self, msg: Odometry): # set state depending on odom callback
        #self.get_logger().info(f"My current position: {msg.pose.pose.position}")
        self.state["x"] = msg.pose.pose.position.x
        self.state["y"] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.state["theta"] = math.atan2(siny_cosp, cosy_cosp)

        if not self.initial_pos_stored:
            self.initial_pos['x'] = self.state['x']
            self.initial_pos['y'] = self.state['y']
            self.initial_pos_stored = True

    def f_s(self):
        return self.INPUT * self.u

    def Du(self):# EFFECT OF THE SPRINGS
        # PE = 1/2 * k * dx^2
        # d PE = k * dx
        if not self.neighbours_positions:
            return 0.0, 0.0
        sum_x, sum_y = 0.0, 0.0
        # print statement of these makes it look
        #self.get_logger().info(f"[DEBUG] NEIGHBOUR POSITIONS: {self.neighbours_positions.items()}")
        for topic, (nx, ny) in self.neighbours_positions.items():
            k = self.neighbour_randoms.get(topic, 1.0) # spring stiffness
            dx, dy = self.state['x'] - nx, self.state['y'] - ny
            d = np.sqrt(dx**2 + dy**2)
            energy_x = k * (self.neighbours_rest_spring[topic] - d) * dx / d
            energy_y = k * (self.neighbours_rest_spring[topic] - d) * dy / d
            sum_x += energy_x
            sum_y += energy_y

        # add spring to own initial position!
        k = self.neighbour_randoms.get(self.name, 1.0) # spring stiffness
        dx, dy = self.state['x'] - self.initial_pos['x'], self.state['y'] - self.initial_pos['y']
        energy_x = k * (self.neighbours_rest_spring[topic] - d) * dx / d
        energy_y = k * (self.neighbours_rest_spring[topic] - d) * dy / d
        sum_x += energy_x
        sum_y += energy_y

        if not MUTE:
            self.get_logger().info(f"sum x: {sum_x} and y: {sum_y} before collision force ") 

        sum_x += self.collision_force['x'] # collision impact
        sum_y += self.collision_force['y']

        if not MUTE:
            self.get_logger().info(f"sum x: {sum_x} and y: {sum_y} after collision force ")  

        return sum_x, sum_y

    def update(self):
        if not MUTE:
            self.get_logger().info(f"[DEBUG] input value {self.u}")
        if not self.anchor: # none of this will change if youre an anchor
            # Angular velocity
            #self.get_logger().info(f"[DEBUG] STARTED UPDATE")
            dw = -(ZETA / self.J) * self.state["w"]

            self.state["w"] += DT * dw

            # Linear velocity
            Dx, Dy = self.Du()
            ds = (Dx * math.cos(self.state["theta"]) +
                Dy * math.sin(self.state["theta"]) +
                self.f_s() - self.BETA * self.state["s"]) / self.M
            
            self.state["s"] += DT * ds

            # additional constraints:
            self.state["s"] = np.clip(self.state['s'], -MAX_SPEED, MAX_SPEED)
            self.state["w"] = np.clip(self.state['w'], -MAX_ANG, MAX_ANG)

            # Publish velocity commands
            cmd = Twist()
            cmd.linear.x = self.state['s']
            cmd.angular.z = self.state['w']
            self.cmd_vel_pub.publish(cmd)
            #self.get_logger().info(f"[DEBUG] Published to cmd_vel: linear.x={cmd.linear.x:.2f}, angular.z={cmd.angular.z:.2f}")

        # Publish position - to odometry filtered?
        msg = Point()
        msg.x, msg.y = self.state['x'], self.state['y']
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    import sys
    name = sys.argv[1]
    anchor = sys.argv[2].lower() == 'true'
    placeholder_robot_names = sys.argv[3].split(',')
    print(f'[DEBUG] robot_names: {placeholder_robot_names}') # blank

    node = DotNode(name=name, placeholder_names=placeholder_robot_names, anchor=anchor)
    rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()



if __name__ == '__main__':
    main()
