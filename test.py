import argparse
from collections import deque, namedtuple
import random
import numpy as np
import carla
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import cv2
import sys
from PIL import Image
import csv


"""
Global Variable List
"""
sim_time = 0
start_time = 0
num_ep = 0
reward_num = 0

"""
Replay buffer class
"""
NUM_ACTIONS = 45


# Carla Client attribute
client = carla.Client("localhost", 2000)
client.set_timeout(5.0)  # Set a timeout in seconds for client connection
print("Client established")


class DuelingDDQN(nn.Module):
    """
    Dueling Double Deep Q-Network (DDQN) model for reinforcement learning.

    This neural network model predicts Q-values for each action given a state.

    Args:
        action_dim (int): Dimensionality of the action space.
        image_dim (tuple): Dimensions of the input image (height, width).

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        pool1 (nn.MaxPool2d): First max pooling layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool2 (nn.MaxPool2d): Second max pooling layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        pool3 (nn.MaxPool2d): Third max pooling layer.
        flatten_size (int): Size of the flattened output of the final convolutional layer.
        fc1 (nn.Linear): Fully connected layer.
        value_stream (nn.Linear): Linear layer for the value stream.
        advantage_stream (nn.Linear): Linear layer for the advantage stream.
    """
    
    def __init__(self, action_dim, image_dim=(480, 640)):
        super(DuelingDDQN, self).__init__()
        # Convolutional and pooling layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten the output of the final convolutional layer
        self.flatten_size = self._get_conv_output((3, image_dim[0], image_dim[1]))

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)

        # State Value stream
        self.value_stream = nn.Linear(512, 1)

        # Advantage stream
        self.advantage_stream = nn.Linear(512, action_dim)

    def _get_conv_output(self, shape):
        """
        Compute the size of the flattened output of the final convolutional layer.

        Args:
            shape (tuple): Shape of the input tensor (channels, height, width).

        Returns:
            int: Size of the flattened output.
        """
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv1(input)
            output = self.pool1(output)
            output = self.conv2(output)
            output = self.pool2(output)
            output = self.conv3(output)
            output = self.pool3(output)
            return int(np.prod(output.size()))

    def forward(self, state):
        """
        Forward pass of the neural network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Predicted Q-values for each action.
        """
        # Convert state to float and scale if necessary
        state = state.float() / 255.0  # Scale images to [0, 1]

        x = F.relu(self.pool1(self.conv1(state)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))

        # Flatten and pass through fully connected layer
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # Value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine to get Q-values

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        #      print(f'Shapes of network, Value{value.shape}, advantage{advantage.shape}, q_values{q_values.shape}')
        return q_values


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = DuelingDDQN(NUM_ACTIONS).to(device)
target_network = deepcopy(network).to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)
loss_fn = nn.SmoothL1Loss()  # huber loss


class ReplayBuffer:
    """
    Replay buffer for experience replay in reinforcement learning.

    This buffer stores experiences and provides methods for storing,
    sampling, and retrieving experiences for training.

    Args:
        capacity (int): Maximum capacity of the replay buffer.

    Attributes:
        buffer (deque): Deque containing the stored experiences.
    """
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer with a given capacity.

        Args:
            capacity (int): Maximum capacity of the replay buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        """
        Store a new experience in the replay buffer.

        Args:
            experience: The experience to be stored in the buffer.
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            list: A list containing the sampled experiences.
        """
        return random.sample(self.buffer, batch_size)

    def size(self):
        """
        Get the current size of the replay buffer.

        Returns:
            int: The current number of experiences stored in the buffer.
        """    
        return len(self.buffer)


class HUD:
    """
    Heads-Up Display (HUD) for visualizing information on camera images.

    This class manages the HUD elements such as speed, throttle, steer,
    heading, location, collision, and nearby vehicles, and provides methods
    for updating and rendering these elements on camera images.

    Args:
        width (int): Width of the camera image.
        height (int): Height of the camera image.

    Attributes:
        dim (tuple): Dimension of the camera image (width, height).
        font: Font used for rendering text on the HUD.
        font_scale (float): Scale factor for adjusting font size.
        font_color (tuple): Color of the text rendered on the HUD.
        line_height (int): Height of each line of text.
        x_offset (int): Horizontal offset for positioning HUD elements.
        y_offset (int): Vertical offset for positioning HUD elements.
        speed (float): Current speed of the vehicle.
        throttle (float): Current throttle value.
        steer (float): Current steering angle.
        heading (str): Current heading of the vehicle.
        location (str): Current location of the vehicle.
        collision (list): List of collision information.
        nearby_vehicles (list): List of nearby vehicles.
    """
    
    def __init__(self, width, height):
        self.dim = (width, height)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_color = (255, 255, 255)
        self.line_height = 20
        self.x_offset = 10
        self.y_offset = 20
        self.speed = 0
        self.throttle = 0
        self.steer = 0
        self.heading = ""
        self.location = ""
        self.collision = []
        self.nearby_vehicles = []

    def update(
        self, speed, throttle, steer, heading, location, collision, nearby_vehicles
    ):
        """
        Update the HUD with new information.

        Args:
            speed (float): Current speed of the vehicle.
            throttle (float): Current throttle value.
            steer (float): Current steering angle.
            heading (str): Current heading of the vehicle.
            location (str): Current location of the vehicle.
            collision (list): List of collision information.
            nearby_vehicles (list): List of nearby vehicles.
        """
        self.speed = speed
        self.throttle = throttle
        self.steer = steer
        self.heading = heading
        self.location = location
        self.collision = collision
        self.nearby_vehicles = nearby_vehicles

    def tick(self, camera_image):
        """
        Render HUD elements on the camera image and return the image.

        Args:
            camera_image: The camera image on which HUD elements are rendered.

        Returns:
            np.array: The camera image with HUD elements rendered.
        """
        # Create a blank HUD image
        hud_image = np.zeros((self.dim[1], self.dim[0], 3), dtype=np.uint8)

        # Add HUD elements
        cv2.putText(
            hud_image,
            f"Speed: {self.speed:.2f} m/s",
            (10, 40),
            self.font,
            self.font_scale,
            self.font_color,
            1,
        )
        cv2.putText(
            hud_image,
            f"Throttle: {self.throttle:.2f}",
            (10, 60),
            self.font,
            self.font_scale,
            self.font_color,
            1,
        )
        cv2.putText(
            hud_image,
            f"Steer: {self.steer:.2f}",
            (10, 80),
            self.font,
            self.font_scale,
            self.font_color,
            1,
        )
        cv2.putText(
            hud_image,
            f"Heading: {self.heading}",
            (10, 100),
            self.font,
            self.font_scale,
            self.font_color,
            1,
        )
        cv2.putText(
            hud_image,
            f"Location: {self.location}",
            (10, 120),
            self.font,
            self.font_scale,
            self.font_color,
            1,
        )
        cv2.putText(
            hud_image,
            "Collision:",
            (10, 140),
            self.font,
            self.font_scale,
            self.font_color,
            1,
        )
        for i, value in enumerate(self.collision):
            cv2.putText(
                hud_image,
                f"{i}: {value:.2f}",
                (10, 160 + i * 20),
                self.font,
                self.font_scale,
                self.font_color,
                1,
            )
        cv2.putText(
            hud_image,
            f"Nearby vehicles:",
            (10, 160 + len(self.collision) * 20),
            self.font,
            self.font_scale,
            self.font_color,
            1,
        )
        for i, vehicle in enumerate(self.nearby_vehicles):
            cv2.putText(
                hud_image,
                f"{i}: {vehicle}",
                (10, 180 + (len(self.collision) + i) * 20),
                self.font,
                self.font_scale,
                self.font_color,
                1,
            )

        # Overlay HUD image onto camera image
        camera_image_with_hud = cv2.addWeighted(camera_image, 1, hud_image, 0.5, 0)

        return camera_image_with_hud

# visualize delete
def draw_spawn_points(world, spawn_points, duration=10.0):
    for idx, spawn_point in enumerate(spawn_points):
        location = spawn_point.location
        world.debug.draw_string(
            location,
            f"SP {idx}",
            draw_shadow=False,
            color=carla.Color(255, 0, 0),  # Red color for visibility
            life_time=duration,
            persistent_lines=True,
        )
        # Draw a small vertical line to mark the exact point
        world.debug.draw_line(
            location,
            location + carla.Location(z=2.0),  # Draw a line upwards
            thickness=0.1,
            color=carla.Color(0, 255, 0),  # Green line
            life_time=duration,
        )


class Environment:
    """
    Class representing the environment for the simulation.

    Args:
        carla_client: The Carla client instance.
        car_config: Configuration for the vehicle.
        sensor_config: Configuration for the sensors.
        reward_function: The reward function to use.
        map: The map to load (default is 0).
        spawn_index: The spawn index for the vehicle.
        random: Whether to use random spawning.
    """
    
    def __init__(
        self,
        carla_client,
        car_config,
        sensor_config,
        reward_function,
        map=0,
        spawn_index=None,
        random=False,
    ):
        # Connecting to Carla Client
        self.client = carla_client
        self.world = self.client.get_world()



        self.traffic_manager = self.client.get_trafficmanager(8000)  # Use port 8000 for the Traffic Manager
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)  # Maintain a minimum distance


        # if loading specifc map
        if map != 0:
            self.world = self.client.load_world(map)
        """ This portion can be moved to env.reset
         #delete what we created, eg. vehicles and sensors
        actor_list = self.world.get_actors()
        vehicle_and_sensor_ids = [actor.id for actor in actor_list if (('vehicle' in  actor.type_id) or ('sensor' in actor.type_id))]
        for id in vehicle_and_sensor_ids:   #delete all vehicles and cameras
            created_actor = self.world.get_actor(id)
            created_actor.destroy()
            print("Deleted", created_actor)
        """
        ## Setting environment attributes
        self.car_config = car_config
        self.random = random
        self.sensor_config = sensor_config
        self.rf = int(reward_function[0])
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bps = [
            bp
            for bp in self.blueprint_library.filter("vehicle")
            if bp.has_attribute("number_of_wheels")
        ]
        self.vehicle_bp = self.vehicle_bps[0]
        self.camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.camera_bp.set_attribute("image_size_x", str(sensor_config["image_size_x"]))
        self.camera_bp.set_attribute("image_size_y", str(sensor_config["image_size_y"]))
        self.camera_bp.set_attribute("fov", str(sensor_config["fov"]))

        """ This portion can be moved to env.reset
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        #available spawn points:
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = random.choice(spawn_points)
        # adding vehicle to self
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
        # spawned vehicle in simulator

        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)

        self.distance = 0
        self.prev_xy = np.zeros((2, ))
        """
        # Action space is now defined in terms of throttle and steer instead of curvature and speed.
        throttle_range = np.linspace(0, 0.5, 5)
        steer_range = np.linspace(-0.25, 0.25, 9)
       # print(steer_range)
        self.action_space = np.array(
            np.meshgrid(throttle_range, steer_range)
        ).T.reshape(-1, 2)
        self.spawn_point = None
        if spawn_index is not None:
            self.spawn_point = self.world.get_map().get_spawn_points()[spawn_index]

        # self.camera.listen(lambda data: self.process_image(data))

        # Initialize HUD
        self.hud = HUD(sensor_config["image_size_x"], sensor_config["image_size_y"])
    def detect_unsafe_lane_change(self):    # change
        """
        Detects if the agent's lane change behavior is unsafe.
        Considers factors like abrupt steering, proximity to other vehicles, and road markings.
        """
        # Get current vehicle information
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation.yaw

        # Get the map and waypoint
        map = self.world.get_map()
        waypoint = map.get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        # Check if the vehicle is crossing solid lane markings (unsafe)
        left_marking = waypoint.left_lane_marking
        right_marking = waypoint.right_lane_marking
        unsafe_crossing = (
            (left_marking and left_marking.type == carla.LaneMarkingType.Solid)
            or (right_marking and right_marking.type == carla.LaneMarkingType.Solid)
        )

        # Check if steering input is too high for an abrupt lane change
        excessive_steering = abs(self.steer) > 0.5  # Example threshold

        # Check proximity to other vehicles
        nearby_vehicles = self.world.get_actors().filter("vehicle.*")
        for other_vehicle in nearby_vehicles:
            if other_vehicle.id == self.vehicle.id:
                continue  # Skip self
            other_location = other_vehicle.get_transform().location
            distance = vehicle_location.distance(other_location)
            if distance < 5.0:  # Threshold for unsafe proximity
                unsafe_proximity = True
                break
        else:
            unsafe_proximity = False

        # Combine conditions
        unsafe_lane_change = unsafe_crossing or excessive_steering or unsafe_proximity

        return unsafe_lane_change

    def overtake_successful(self):
        """
        Determines if the agent has successfully overtaken another vehicle.
        Conditions for success:
        - The agent moves past another vehicle in the same direction.
        - No collision occurs during the overtaking maneuver.
        - The agent maintains appropriate lane alignment post-overtaking.
        """
        # Get the current vehicle location
        vehicle_location = self.vehicle.get_transform().location

        # Get nearby vehicles
        nearby_vehicles = self.world.get_actors().filter("vehicle.*")
        overtaken_vehicle = None

        for other_vehicle in nearby_vehicles:
            if other_vehicle.id == self.vehicle.id:
                continue  # Skip self

            other_location = other_vehicle.get_transform().location
            relative_position = vehicle_location.x - other_location.x

            # Check if the vehicle has overtaken another vehicle
            if 0 < relative_position < 10:  # Example threshold for being ahead
                overtaken_vehicle = other_vehicle
                break

        # Conditions for successful overtaking
        no_collision = not self.collision_detected
        maintaining_lane = not self.detect_unsafe_lane_change()

        # Successful overtaking if vehicle has moved ahead safely
        overtake_successful = overtaken_vehicle is not None and no_collision and maintaining_lane
        return overtake_successful

    def check_excessively_conservative(self):
        """
        Determines if the agent is being excessively conservative.
        Conditions for excessive conservatism:
        - Driving significantly below the speed limit without reason.
        - Failing to overtake slower vehicles when it's safe to do so.
        - Unnecessary stops or hesitation in clear scenarios.
        """
        # Get the vehicle speed
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # Convert to scalar speed (m/s)

        # Get the speed limit from the waypoint
        waypoint = self.world.get_map().get_waypoint(self.vehicle.get_transform().location)
        speed_limit = 70

        # Get nearby vehicles
        nearby_vehicles = self.world.get_actors().filter("vehicle.*")
        ahead_vehicle = None

        for other_vehicle in nearby_vehicles:
            if other_vehicle.id == self.vehicle.id:
                continue  # Skip self

            # Check if another vehicle is directly ahead within a certain range
            other_location = other_vehicle.get_transform().location
            distance = self.vehicle.get_transform().location.distance(other_location)
            if distance < 20 and other_location.x > self.vehicle.get_transform().location.x:
                ahead_vehicle = other_vehicle
                break

        # Conditions for excessive conservatism
        driving_too_slow = speed < 0.6 * speed_limit  # Example: <60% of speed limit
        not_overtaking = ahead_vehicle is not None and not self.check_overtake_successful()

        excessively_conservative = driving_too_slow or not_overtaking
        return excessively_conservative

    def on_collision(self, event):
        """
        Callback function for collision events.
        """
        self.collision_detected = True

    def reset(self):  # reset is to reset world?
        """
        Reset the environment.
        """
        # Spawn or respawn the vehicle at a random location
        # delete what we created, eg. vehicles and sensors
        actor_list = self.world.get_actors()

        # Identify IDs of vehicles and sensors to be deleted
        vehicle_and_sensor_ids = [
            actor.id
            for actor in actor_list
            if "vehicle" in actor.type_id or "sensor" in actor.type_id
        ]

        # Iterate over the list of IDs to delete each actor
        for actor_id in vehicle_and_sensor_ids:
            # Attempt to get the actor by ID
            actor_to_delete = self.world.get_actor(actor_id)
            if actor_to_delete is not None:
                # If the actor exists, delete it
                actor_to_delete.destroy()
                print(
                    "Deleted:", actor_to_delete.type_id, "with ID:", actor_to_delete.id
                )
            else:
                # If the actor doesn't exist, print a message (optional)
                print("Actor with ID", actor_id, "not found.")

        if self.spawn_point is None or self.random:
            spawn_points = self.world.get_map().get_spawn_points()
            self.spawn_point = random.choice(spawn_points)
            print(f"spawn index: {spawn_points.index(self.spawn_point)}")
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
        self.vehicle.set_autopilot(False)
        self.vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))

        # Attach the camera sensor
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )  # camera offset relative to cars origin,
        # changeable, depending where the camera actually is
        self.camera = self.world.spawn_actor(
            self.camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(lambda data: self.process_image(data))
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_detected = False
        self.collision_sensor.listen(lambda event: self.on_collision(event))

        self.distance = 0
        self.prev_xy = np.array(
            [self.vehicle.get_location().x, self.vehicle.get_location().y]
        )
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        # waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        # Start collecting data
        self.image = None
        while self.image is None:
            self.world.tick()
        return self.image

    def process_image(self, image):
        """
        Process the image received from the camera sensor.
        """
        i = np.array(image.raw_data)
        i2 = i.reshape(
            (self.sensor_config["image_size_y"], self.sensor_config["image_size_x"], 4)
        )
        self.image = i2[:, :, :3]
        image_array_copy = self.image.copy()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_color = (255, 255, 255)
        elapsed_since_last_iteration = time.time() - start_time
        self.time = elapsed_since_last_iteration
        self.episode = num_ep
        self.reward_number = reward_num

        # # vehicle control
        """
        ###
        control = self.vehicle.get_control()

        # # throttle bar
        throttle_bar_length = int(100 * control.throttle)
        throttle_bar_x = 80
        throttle_bar_y = 150
        # # unfilled rectangle
        cv2.rectangle(image_array_copy, (throttle_bar_x, throttle_bar_y), (throttle_bar_x + 100, throttle_bar_y + 10), (255, 255, 255), 1)
        # # throttle fill
        throttle_color = (int(255 * control.throttle), int(255 * (1 - control.throttle)), 0)
        cv2.rectangle(image_array_copy, (throttle_bar_x + 1, throttle_bar_y + 1), (throttle_bar_x + throttle_bar_length, throttle_bar_y + 9), throttle_color, -1)

        # # steer bar
        steer_bar_length = int(50 * (control.steer + 1))  # Adjust multiplier as needed
        steer_bar_x = 80
        steer_bar_y = 170
        cv2.rectangle(image_array_copy, (steer_bar_x, steer_bar_y), (steer_bar_x + 100, steer_bar_y + 10), (255, 255, 255), 1)
        # # Draw slider for steer value
        slider_x = steer_bar_x + int(100 * (control.steer + 1) / 2)
        slider_y = steer_bar_y
        cv2.rectangle(image_array_copy, (slider_x - 3, slider_y), (slider_x + 3, slider_y + 9), (255, 255, 255), -1)

        # # Calculate the speed (magnitude of velocity)
        velocity = self.vehicle.get_velocity()
        speed = velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        speed = speed ** 0.5

        # # display location
        location = self.vehicle.get_location()
        formatted_location = "({:.2f}, {:.2f})".format(location.x, location.y)

        cv2.putText(image_array_copy, f'Simulation Time: {self.time:.2f} s', (10, 40), self.font, 0.5, self.font_color)
        cv2.putText(image_array_copy, f'Reward Function: {self.reward_number}', (10, 60), self.font, 0.5, self.font_color)
        cv2.putText(image_array_copy, f'Episode Number: {self.episode}', (10, 80), self.font, 0.5, self.font_color)
        cv2.putText(image_array_copy, f'Speed: {speed:.2f} m/s', (10, 100), self.font, 0.5, self.font_color)
        cv2.putText(image_array_copy, f'Location: {formatted_location}', (10, 120), self.font, 0.5, self.font_color)
        cv2.putText(image_array_copy, "Throttle:", (10, 160), self.font, self.font_scale, self.font_color)
        cv2.putText(image_array_copy, "Steer:", (10, 180), self.font, self.font_scale, self.font_color)

        cv2.imshow("Camera View", image_array_copy)
        cv2.waitKey(5)
        """
        ###

    def step(self, action):
        """
        Take a step in the environment based on the given action.

        Args:
            action: The action to take.
        """
        self.throttle, self.steer = action
        #  print(self.action_space)
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=self.throttle, steer=self.steer)
        )

        self.world.tick()

        # Compute the distance traveled since the last step
        current_location = self.vehicle.get_location()
        current_xy = np.array([current_location.x, current_location.y])
        dd = np.linalg.norm(current_xy - self.prev_xy)
        self.distance += dd

        info = {}

        # getting info data

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation.yaw
        map = self.world.get_map()
        waypoint = map.get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        vehicle_rotation_radians = math.radians(vehicle_rotation)
        vehicle_rotation_radians = (vehicle_rotation_radians + np.pi) % (
            2 * np.pi
        ) - np.pi

        road_direction = waypoint.transform.rotation.yaw
        road_direction_radians = math.radians(road_direction)
        theta = abs(vehicle_rotation_radians - road_direction_radians) % (2 * np.pi)
        if theta > np.pi:
            theta = 2 * np.pi - theta
        going_opposite_direction = theta > np.pi / 2

        road_half_width = waypoint.lane_width / 2.0

        center_of_lane = waypoint.transform.location
        distance_from_center = vehicle_location.distance(center_of_lane)

        not_near_center = distance_from_center > road_half_width / 1.5
        done = not_near_center or going_opposite_direction or self.collision_detected

        current_xy = np.array([vehicle_location.x, vehicle_location.y])
        dd = np.linalg.norm(current_xy - self.prev_xy)

        Py = distance_from_center

        velocity = self.vehicle.get_velocity()
        speed = velocity.x**2 + velocity.y**2 + velocity.z**2
        speed = speed**0.5

        info["angle"] = math.cos(theta)
        info["lane_deviation"] = Py
        info["collision"] = 1 if self.collision_detected else 0
        info["speed"] = speed

        # Calculate reward based on the chosen reward function
        if self.rf == 1:
            reward, done = self.reward_1()
        elif self.rf == 2:
            reward, done = self.reward_2()
        elif self.rf == 3:
            reward, done = self.reward_3()
        elif self.rf == 4:
            reward, done, theta, Py = self.reward_4()
            info["angle"] = math.cos(theta)
            info["lane_deviation"] = Py
            info["collision"] = 1 if self.collision_detected else 0

        self.prev_xy = current_xy
        return self.image, reward, done, info

    def reward_1(self):
        """
        Discrete reward function for CARLA:
        - Penalize the agent heavily for getting out of lane.
        - Penalize for exceeding max rotation.
        - Penalize for not being centered on the road.
        - Penalize heavily if the vehicle is going in the opposite direction of the road.
        
        Returns:
            reward (float): The reward value.
            done (bool): Whether the episode is done.
        """

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation.yaw
    #    print("Vehicle location is", vehicle_location.x, vehicle_location.y)
        #     print("Vehicle Rotation is", vehicle_rotation)
        # Convert yaw to radians and normalize between -pi and pi
        vehicle_rotation_radians = math.radians(vehicle_rotation)
        vehicle_rotation_radians = (vehicle_rotation_radians + np.pi) % (
            2 * np.pi
        ) - np.pi

        # Maximal rotation (yaw angle) allowed
        maximal_rotation = np.pi / 10
        exceed_max_rotation = np.abs(vehicle_rotation_radians) > maximal_rotation

        # Getting the vehicle's lane information
        map = self.world.get_map()
        waypoint = map.get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        #   print("Map is", map)
        # Calculate the heading difference between the vehicle and the road
        road_direction = waypoint.transform.rotation.yaw
        #   print("Road Direction is", road_direction)
        road_direction_radians = math.radians(road_direction)
        #   print("Road Direction radians is", road_direction_radians)
        #   print("Vehicle direction radians is",vehicle_rotation_radians)
        heading_difference = abs(vehicle_rotation_radians - road_direction_radians) % (
            2 * np.pi
        )

        # Heavily penalize if the vehicle is going in the opposite direction (more than 90 degrees away from road direction)
        going_opposite_direction = heading_difference > np.pi / 2

        road_half_width = waypoint.lane_width / 2.0
        center_of_lane = waypoint.transform.location
        distance_from_center = vehicle_location.distance(center_of_lane)
        #     print("Distance from center is", distance_from_center)
        out_of_lane = self.is_vehicle_within_lane() is False
        #   print("Is out of lane?", out_of_lane)
        #   print("Road half width is", road_half_width)
        not_near_center = distance_from_center > road_half_width / 2
        #    print(not_near_center, math.degrees(heading_difference))
        #   print("Previous xy is", self.prev_xy)
        # Determine if the episode should end
        done = not_near_center or going_opposite_direction or self.collision_detected

        #  print("Are we done?", done)
        #  print("Heading distance is", heading_difference)
        # Compute reward based on conditions
        current_xy = np.array([vehicle_location.x, vehicle_location.y])
        reward = 0

        #  print("Current xy is", current_xy)
        #  print("Prev xy is", self.prev_xy)
        if self.collision_detected:
            done = True
            reward = -1000
        elif done:
            reward = (
                -100 if not_near_center else -500
            )  # More severe penalty for going in the opposite direction
        elif exceed_max_rotation:
            reward = -50
        else:
            # Calculate distance moved towards the driving direction since last tick
            dd = np.linalg.norm(current_xy - self.prev_xy)
            reward = (
                dd * 50
            )  # Assuming the simulation has a tick rate where this scaling makes sense

        #   print("Reward from ifelif is", reward)

        reward += (abs(heading_difference)) * -100

        self.prev_xy = current_xy
        self.collision_detected = False
        return reward, done

    def reward_2(self):
        """
        Reward function that does not account for max rotation exceeded
        
        Returns:
            reward (float): The reward value.
            done (bool): Whether the episode is done.
        """

        exceed_max_rotation = np.abs(vehicle_rotation_radians) > maximal_rotation

        # Getting the vehicle's lane information
        map = self.world.get_map()
        waypoint = map.get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        road_half_width = waypoint.lane_width / 2.0

        # Calculate the distance from the center of the lane+
        center_of_lane = waypoint.transform.location
        distance_from_center = vehicle_location.distance(center_of_lane)

        # Determine if the vehicle is out of lane or not near the center
        out_of_lane = self.is_vehicle_within_lane() is False
        not_near_center = distance_from_center > road_half_width / 4
        #print(distance_from_center)

        # Determine if the episode should end
        done = out_of_lane

        # Compute reward based on conditions
        current_xy = np.array([vehicle_location.x, vehicle_location.y])
        dd = np.linalg.norm(current_xy - self.prev_xy)
        reward = 0

        if out_of_lane:
            reward = -100
        else:
            reward = dd * 5

        if not_near_center:
            reward -= 0.5
        else:
            reward += 2

        return reward, done

    def reward_3(self):
        """
        Reward function with a different formulation.
        
        Returns:
            reward (float): The reward value.
            done (bool): Whether the episode is done.
        """
        reward = 0
        done = False
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation.yaw
        map = self.world.get_map()
        waypoint = map.get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        # Convert yaw to radians and normalize between -pi and pi
        vehicle_rotation_radians = math.radians(vehicle_rotation)
        vehicle_rotation_radians = (vehicle_rotation_radians + np.pi) % (
            2 * np.pi
        ) - np.pi

        road_direction = waypoint.transform.rotation.yaw
        road_direction_radians = math.radians(road_direction)
        theta = abs(vehicle_rotation_radians - road_direction_radians) % (2 * np.pi)
        if theta > np.pi:
            theta = 2 * np.pi - theta
        going_opposite_direction = theta > np.pi / 2

        road_half_width = waypoint.lane_width / 2.0

        center_of_lane = waypoint.transform.location
        distance_from_center = vehicle_location.distance(center_of_lane)

        not_near_center = distance_from_center > road_half_width / 1.5
        done = not_near_center or going_opposite_direction or self.collision_detected

        current_xy = np.array([vehicle_location.x, vehicle_location.y])
        dd = np.linalg.norm(current_xy - self.prev_xy)
        Py = distance_from_center
        Wd = waypoint.lane_width / 2.5
        i_fail = 1 if done else 0

        reward = dd + 2 * math.cos(theta) - abs(Py / Wd) - (4 * i_fail)
        print("Theta is", theta)
        print("Reward 3 reward is", reward)
        return reward, done

    def reward_4(self):
        """
        Reward function with additional considerations.
        
        Returns:
            reward (float): The reward value.
            done (bool): Whether the episode is done.
            theta (float): Angle difference between vehicle heading and road direction.
            Py (float): Distance from the center of the lane.
        """
        reward = 0
        done = False
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation.yaw
        map = self.world.get_map()
        waypoint = map.get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        vehicle_rotation_radians = math.radians(vehicle_rotation)
        vehicle_rotation_radians = (vehicle_rotation_radians + np.pi) % (
            2 * np.pi
        ) - np.pi

        road_direction = waypoint.transform.rotation.yaw
        road_direction_radians = math.radians(road_direction)
        theta = abs(vehicle_rotation_radians - road_direction_radians) % (2 * np.pi)
        if theta > np.pi:
            theta = 2 * np.pi - theta
        going_opposite_direction = theta > np.pi / 2

        road_half_width = waypoint.lane_width / 2.0

        center_of_lane = waypoint.transform.location
        distance_from_center = vehicle_location.distance(center_of_lane)

        not_near_center = distance_from_center > road_half_width / 1.5
        done = not_near_center or going_opposite_direction or self.collision_detected

        current_xy = np.array([vehicle_location.x, vehicle_location.y])
        dd = np.linalg.norm(current_xy - self.prev_xy)

        Py = distance_from_center
        Wd = waypoint.lane_width / 2.5
       # print(self.steer)
        i_fail = 1 if distance_from_center > road_half_width / 2.5 else 0
        reward = (
            math.sqrt(dd)
            + (math.cos(theta) - abs(Py / Wd) - (2 * i_fail))
            - 2 * abs(self.steer)
        )

        return reward, done, theta, abs(Py)

    def get_vehicle_direction(self):
        """
        Get the direction vector of the vehicle based on its yaw angle.
        
        Returns:
            carla.Vector3D: The direction vector of the vehicle.
        """
        transform = self.vehicle.get_transform()
        rotation = transform.rotation
        radians = math.radians(rotation.yaw)
        return carla.Vector3D(math.cos(radians), math.sin(radians), 0.0)

    def get_road_direction(self):
        """
        Get the direction vector of the road based on the current waypoint.
        
        Returns:
            carla.Vector3D: The direction vector of the road.
        """
        # This is a simplified example. You'll need to adapt it based on how your road data is structured
        map = self.world.get_map()
        waypoint = map.get_waypoint(self.vehicle.get_location())
        next_waypoint = waypoint.next(1.0)[0]  # Assuming there's a next waypoint
        direction = next_waypoint.transform.location - waypoint.transform.location
        return direction.make_unit_vector()

    def calculate_angle_between_vectors(self, v1, v2):
        """
        Calculate the angle between two vectors.
        
        Args:
            v1 (carla.Vector3D): The first vector.
            v2 (carla.Vector3D): The second vector.
        
        Returns:
            float: The angle between the two vectors in degrees.
        """
        dot_product = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
        magnitude_v1 = math.sqrt(v1.x**2 + v1.y**2 + v1.z**2)  # Magnitude of v1
        magnitude_v2 = math.sqrt(v2.x**2 + v2.y**2 + v2.z**2)  # Magnitude of v2

        # Ensure the magnitude is not zero to avoid division by zero error
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            raise ValueError(
                "One or both vectors have zero magnitude, can't calculate angle."
            )

        # Normalize the dot product by the magnitudes of v1 and v2
        cosine_of_angle = dot_product / (magnitude_v1 * magnitude_v2)

        # Clamp the cosine_of_angle to the domain of acos to avoid domain errors
        cosine_of_angle = max(min(cosine_of_angle, 1), -1)

        angle = math.acos(cosine_of_angle)  # Angle in radians

        return math.degrees(angle)  # Convert the angle to degrees

    def get_lateral_position_error_and_lane_width(self):
        """
        Calculate the lateral position error and lane width of the vehicle.
        
        Returns:
            tuple: A tuple containing lateral position error (float) and lane width (float).
        """
        # Get the vehicle's location
        vehicle_location = self.vehicle.get_location()
        map = self.world.get_map()

        # Get the closest waypoint to the vehicle's location
        closest_waypoint = map.get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        # Calculate the lateral position error
        # This is a simple approximation. For more accuracy, consider the direction of the road
        lateral_position_error = (
            vehicle_location.distance(closest_waypoint.transform.location)
            - closest_waypoint.lane_width / 2
        )

        # Get the lane width
        lane_width = closest_waypoint.lane_width

        return lateral_position_error, lane_width

    def is_vehicle_within_lane(self):
        """
        Check if the vehicle is within the lane boundaries.
        
        Returns:
            bool: True if the vehicle is within the lane, False otherwise.
        """
        # Get the vehicle's location
        map = self.world.get_map()
        vehicle_location = self.vehicle.get_location()

        # Get the closest waypoint to the vehicle's location, considering only driving lanes
        closest_waypoint = map.get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        # Get the transform of the closest waypoint
        waypoint_transform = closest_waypoint.transform

        # Calculate the vector from the waypoint to the vehicle
        vehicle_vector = vehicle_location - waypoint_transform.location
        vehicle_vector = carla.Vector3D(
            vehicle_vector.x, vehicle_vector.y, 0
        )  # Ignore Z component

        # Calculate the forward vector of the waypoint (direction the lane is facing)
        waypoint_forward_vector = waypoint_transform.get_forward_vector()
        waypoint_forward_vector = carla.Vector3D(
            waypoint_forward_vector.x, waypoint_forward_vector.y, 0
        )  # Ignore Z component

        # Calculate the right vector of the waypoint (perpendicular to the forward vector)
        waypoint_right_vector = carla.Vector3D(
            -waypoint_forward_vector.y, waypoint_forward_vector.x, 0
        )

        # Project the vehicle vector onto the waypoint right vector to get the lateral distance from the lane center
        lateral_distance = vehicle_vector.dot(waypoint_right_vector)

        # Check if the absolute value of lateral_distance is less than or equal to half the lane width
        is_within_lane = abs(lateral_distance) <= (closest_waypoint.lane_width / 2)

        return is_within_lane

    def epsilon_greedy_action(self, state, epsilon):
        """
        Perform an epsilon-greedy action selection based on the given state and epsilon value.
        
        Args:
            state (torch.Tensor): The current state representation.
            epsilon (float): The exploration rate.
        
        Returns:
            int: The selected action index.
        """
        # print(f"\tstate.shape = {state.shape}")
        state = state.permute(0, 3, 1, 2)
        prob = np.random.uniform()
        if prob < epsilon:
            self.action_idx = np.random.randint(len(self.action_space))
            return self.action_space[self.action_idx]
        else:
            qs = network.forward(state).cpu().data.numpy()
            self.action_idx = np.argmax(qs)
            return self.action_space[self.action_idx]


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


def optimize_model(memory, batch_size, gamma):
    """
    Optimize the Q-network model using a batch of transitions from the replay memory.

    Args:
        memory (ReplayMemory): The replay memory containing transitions.
        batch_size (int): The size of the batch to sample from the replay memory.
        gamma (float): The discount factor for future rewards.
    """
    # print("__FUNCTION__optimize_model()")
    if memory.size() < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # convert to tensors and move to device
    state_batch = torch.cat([s for s in batch.state]).to(device)
    # action_batch = torch.cat([a for a in batch.action]).to(device)
    action_batch = torch.cat([torch.tensor([a]).to(device) for a in batch.action])
    reward_batch = torch.cat([torch.tensor([r]).to(device) for r in batch.reward])
    next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).to(
        device
    )
    done_batch = torch.cat([torch.tensor([d]).to(device) for d in batch.done])
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)

    # print(f"\tstate_batch.shape = {state_batch.shape}")
    state_batch = state_batch.permute(0, 3, 1, 2)
    next_state_batch = next_state_batch.permute(0, 3, 1, 2)
    # print(f"\tstate_batch.shape = {state_batch.shape}")
    # print(f"\taction_batch.shape = {action_batch.shape}")
    # print(f"\treward_batch.shape = {reward_batch.shape}")
    # print(f"\tnext_state_batch.shape = {next_state_batch.shape}")
    # print(f"\tdone_batch.shape = {done_batch.shape}")
    # Compute Q
    current_q = network(state_batch)
    # print("  __FUNCTION__optimize_model()")
    # print(f"\tcurrent_q.shape = {current_q.shape}")
    # print(f"\taction_batch.unsqueeze(1).shape = {action_batch.unsqueeze(1).shape}")
    # print(f"\taction_batch.unsqueeze(1).long().shape = {action_batch.unsqueeze(1).long().shape}")
    current_q = torch.gather(
        current_q, dim=1, index=action_batch.unsqueeze(1).long()
    ).squeeze(-1)
    # print(f"\tcurrent_q.shape = {current_q.shape}")
    # print(f"\tcurrent_q = {current_q}")

    # print(f"\tlen(next_state_batch) = {len(next_state_batch)}")

    with torch.no_grad():
        # compute target Q
        target_q = torch.full([len(next_state_batch)], 0, dtype=torch.float32)
        # target_q = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.float32)

        for idx in range(len(next_state_batch)):
            reward = reward_batch[idx]
            next_state = next_state_batch[idx].unsqueeze(0)
            done = done_batch[idx]
            if done:
                target_q[idx] = reward
                # print(f"\t\t\treward = {reward}")
            else:
                # print(f"\tstate_batch.shape = {state_batch.shape}")
                # print(f"\t\t\tnext_state.shape = {next_state.shape}")
                q_values = target_network(next_state)
                # print(f"\t\t\tt_q_values.shape = {q_values.shape}")
                max_q = q_values[0][torch.argmax(q_values)]
                target = reward + gamma * max_q
                target_q[idx] = target

    # print(f"\ttarget_q.shape = {target_q.shape}")

    # Compute Huber loss
    loss_q = loss_fn(current_q.to(device), target_q.to(device))

    # Optimize the model
    optimizer.zero_grad()
    loss_q.backward()
    optimizer.step()


def update_plot(rewards, num_steps, lane_deviation, angle, speed):
    """
    Update the training plot with new data.

    Args:
        rewards (list): List of average rewards per episode.
        num_steps (list): List of number of steps per episode.
        lane_deviation (list): List of lane deviation values per episode.
        angle (list): List of angle values per episode.
        speed (list): List of speed values per episode.
    """
    # with open('plot')
    # plt.clf()  just adds blank figure
    plt.figure(figsize=(10, 8))

    # create plots
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(0, len(rewards)), rewards)
    plt.xlabel("Training Episodes")
    plt.ylabel("Average Reward per Episode")
    plt.title("Average Reward")

    # Plot the number of steps per episode
    plt.subplot(4, 1, 2)
    plt.plot(np.arange(0, len(num_steps)), num_steps)
    plt.xlabel("Training Episodes")
    plt.ylabel("Number of Steps per Episode")
    plt.title("Steps per Episode")

    plt.subplot(4, 1, 3)
    y1 = lane_deviation
    y2 = angle
    y3 = speed
    x = len(speed)
    plt.plot(np.arange(0, x), y1, label="Lane Deviation (distance from center)")
    plt.plot(np.arange(0, x), y2, label="Angle (radians)")
    plt.plot(np.arange(0, x), y3, label="Speed (m/s)")
    plt.xlabel("Training Episodes")
    plt.ylabel("Lane Deviation, Angle, Speed")
    plt.title("Lane Deviation, Angle, Speed per Episode")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.scatter(np.arange(0, len(rewards)), rewards, color="blue")
    coeffs = np.polyfit(np.arange(len(rewards)), rewards, 1)
    p = np.poly1d(coeffs)
    plt.plot(np.arange(len(rewards)), p(np.arange(len(rewards))), "r--")
    plt.xlabel("Training Episodes")
    plt.ylabel("Average Reward per Episode")
    plt.title(
        "Average Reward (Corr: {:.2f})".format(
            np.corrcoef(rewards, p(np.arange(len(rewards))))[0, 1]
        )
    )

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Running Main")
    parser = argparse.ArgumentParser(
        description="Run the simulator with random actions"
    )
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        help="version number for model naming",
        required=False,
    )
    parser.add_argument(
        "--operation", type=str, nargs="+", help="Load or New or Tune", required=True
    )
    parser.add_argument(
        "--save-path",
        type=str,
        nargs="+",
        help="Path to the saved model state",
        required=False,
    )
    # reward function
    parser.add_argument(
        "--reward-function",
        type=str,
        nargs="+",
        help="1 or 2 or 3 or 4 or",
        required=True,
    )
    parser.add_argument(
        "--map",
        type=str,
        nargs="+",
        help="Specify CARLA map: (Town01, ...  Town07)",
        required=False,
    )
    parser.add_argument(
        "--epsilon-decrement", type=str, nargs="+", help="Epsilon", required=False
    )
    parser.add_argument(
        "--num-episodes",
        type=str,
        nargs="+",
        help="Number of episodes for training",
        required=False,
    )
    parser.add_argument(
        "--max-steps",
        type=str,
        nargs="+",
        help="Maximum number of steps per episode",
        required=False,
    )
    parser.add_argument(
        "--random-spawn",
        type=str,
        nargs=1,
        help="Vehicle spawn location random? (True/False)",
        required=False,
    )
    args = parser.parse_args()

    if not args.operation:
        print("Operation argument is required")
        raise ValueError(
            "Operation argument is required. Use '--operation Load' or '--operation New'."
        )

    # configuring environment
    car_config = 0  # likely not needed

    sensor_config = {  # default sensor configuration
        "image_size_x": 640,  # Width of the image in pixels
        "image_size_y": 480,  # Height of the image in pixels
        "fov": 90,  # Field of view in degrees
    }

    spawn_points = [67, 99, 52, 56, 44, 5, 100, 40]
    spawn_point = random.choice(spawn_points)
    print("Spawn Point:", spawn_point)
    map = 0  # default map
    if args.map:  # specifed map is chosen
        map = args.map[0]

    random_spawn = True  # default random value
    if args.random_spawn:
        if args.random_spawn[0] == "False":
            random_spawn = False

    env = Environment(
        client,
        car_config,
        sensor_config,
        args.reward_function,
        map,
        19,
        random=random_spawn,
    )

    print("Arguments received:")
    print("Version:", args.version)
    print("Operation:", args.operation)
    print("Save Path:", args.save_path)
    print("Reward Function:", args.reward_function)
    print("Map:", args.map)
    print("Epsilon Decrement:", args.epsilon_decrement)
    print("Number of Episodes:", args.num_episodes)
    print("Max Steps per Episode:", args.max_steps)
    print("Random Vehicle Spawn:", args.random_spawn)

    # initialize HUD
    hud = HUD(sensor_config["image_size_x"], sensor_config["image_size_y"])
    if args.operation[0].lower() == "new" or args.operation[0].lower() == "tune":
        """
        Initializing hyper-parameters and beginning the training loop


        """

        if args.operation[0].lower() == "tune":
            network.load_state_dict(torch.load(args.save_path[0]))
            target_network = deepcopy(network)

        replay_buffer = ReplayBuffer(10000)
        batch_size = 64
        gamma = 0.99
        epsilon_start = 1
        epsilon_end = 0.01
        epsilon_decay = 0.993
        epsilon_decrement = 0.001
        num_episodes = 2000
        max_num_steps = 400
        if args.epsilon_decrement:
            epsilon_decrement = float(args.epsilon_decrement[0])  # default value 0.005
        if args.num_episodes:
            num_episodes = int(args.num_episodes[0])  # default 600
        if args.max_steps:
            max_num_steps = int(args.max_steps[0])  # default 300

        reward_num = args.reward_function[0]

        target_update = 10  # Update target network every 10 episodes

        best_dict_reward = -1e10

        # per episode
        rewards = np.array([])
        num_steps = np.array([])

        epsilon = epsilon_start
        start_time = time.time()
        # opening file to append data
        file = open("plot_data.csv", "w", newline="")
        writer = csv.writer(file)
        file2 = open("step_plot.csv", "w", newline="")
        writer2 = csv.writer(file2)
        writer2.writerow(["lane_dev_avg", "angle_avg", "speed_avg"])
        lane_deviations = []
        speeds = []
        angles = []
        for episode in range(num_episodes):
            ep_deviation = []
            ep_angles = []
            ep_speed = []
            num_ep = episode
            state = env.reset()
            elapsed_since_last_iteration = time.time() - start_time
            start_time = time.time()

            # print(f"main, state.shape after reset = {state.shape.app}")
            # print(state)
            #     display.reset()
            total_reward = 0
            done = False
            step = 0

            while step < max_num_steps and not done:
                # Convert state to the appropriate format and move to device
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

                # Select action using epsilon greedy policy
                action = env.epsilon_greedy_action(state_tensor, epsilon)
                next_state, reward, done, info = env.step(action)  # data here
                # next_state = next_state
                ep_deviation.append(info["lane_deviation"])
                ep_angles.append(info["angle"])
                ep_speed.append(info["speed"])
                # Convert next_state to tensor and move to device
                next_state_tensor = (
                    torch.from_numpy(next_state).unsqueeze(0).to(device)
                    if next_state is not None
                    else None
                )

                replay_buffer.store(
                    (state_tensor, env.action_idx, reward, next_state_tensor, done)
                )

                state = next_state
                total_reward += reward

                # vis_img = display.render()

                # Optimize the model if the replay buffer has enough samples
                optimize_model(replay_buffer, batch_size, gamma)

                if step % target_update == 0 or done:
                    target_network.load_state_dict(network.state_dict())

                step += 1
                # cv2.imshow(f'Car Agent in Episode {episode}', vis_img[:, :, ::-1])
                # cv2.waitKey(5)

                # hud.update(env.vehicle.get_velocity().x, throttle, steer)

                # Get camera image
                camera_image = env.image  # Assumss

                # Display HUD and camera view
                # camera_image_with_hud = hud.tick(camera_image)
                # cv2.imshow("Camera View with HUD", camera_image_with_hud)
                # cv2.waitKey(1)
            # while loop ends
            lane_dev_avg = np.mean(ep_deviation)
            angle_avg = np.mean(ep_angles)
            speed_avg = np.mean(ep_speed)
            lane_deviations.append(lane_dev_avg)
            angles.append(angle_avg)
            speeds.append(speed_avg)

            # write this data to a file for frontend use

            rewards = np.append(rewards, total_reward / step)
            num_steps = np.append(num_steps, step)
            data = [float(total_reward) / float(step), float(step)]
            data2 = [lane_dev_avg, angle_avg, speed_avg]
            #  print("data to be written", data)
            writer.writerow(data)
            writer2.writerow(data2)
            file.flush()
            file2.flush()
            if total_reward > best_dict_reward:
                print("Saving new best")
                torch.save(
                    target_network.state_dict(),
                    "saves/v" + args.version[0] + "_best_dqn_network_nn_model.pth",
                )
                best_dict_reward = total_reward

            print(
                f"Episode {episode}: Total Reward: {total_reward}, Epsilon: {epsilon}, NumSteps: {step}"
            )

            # Update epsilon
            #     epsilon = max(epsilon_end, epsilon_decay * epsilon)
            epsilon = max(epsilon_end, epsilon - epsilon_decrement)

        # Save the model's state dictionary
        torch.save(
            target_network.state_dict(),
            "saves/v" + args.version[0] + "_final_dqn_network_nn_model.pth",
        )
        file.close()
        file2.close()

        # for loop ends

        eps = np.arange(0, num_episodes)
        print(f"rewards = {rewards}")
        print(f"num_steps = {num_steps}")

        # update plot for frontend
        update_plot(rewards, num_steps, lane_deviations, angles, speeds)
        #   display.render()
        plt.show()

        # Create a line graph
        plt.plot(eps, rewards)
        # Add labels and a title
        plt.xlabel("Training Episodes")
        plt.ylabel("Average Reward per Episode")
        plt.title("Average Reward")

        # Display the plot
        plt.show()

        plt.plot(eps, num_steps)
        plt.xlabel("Training Episodes")
        plt.ylabel("Number of Steps per Episode")
        plt.title("Steps per Episode")
        # Display the plot
        plt.show()
    elif args.operation[0].lower() == "load":
        print(f"Loading model from {args.save_path[0]}")
        network = DuelingDDQN(NUM_ACTIONS).to(device)
        network.load_state_dict(torch.load(args.save_path[0]))
        network.eval()
        num_episodes = int(args.num_episodes[0])
        total_rewards = []
        angles = []
        lane_deviation = []
        num_collisions = 0
        for episode in range(int(args.num_episodes[0])):
            state = env.reset()
            done = False
            total_reward = 0
            ep_angles = []
            ep_deviation = []
            collided = 0
            while not done:
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                action = env.epsilon_greedy_action(state_tensor, 0.1)
                state, reward, done, info = env.step(action)
                ep_angles.append(info["angle"])
                ep_deviation.append(info["lane_deviation"])
                if collided:
                    collided = 1
                total_reward += reward
            total_rewards.append(total_reward)
            num_collisions += collided
            angles.append(np.mean(ep_angles))
            lane_deviation.append(np.mean(ep_deviation))
            print(
                f"Episode {episode + 1}: Total Reward = {total_reward}, Angle deviation = {np.mean(ep_angles)}, Lane deviation = {np.mean(ep_deviation)}, Collided = {collided == 1}"
            )

        average_reward = sum(total_rewards) / len(total_rewards)
        average_angle_dev = sum(angles) / len(angles)
        average_lane_dev = sum(lane_deviation) / len(lane_deviation)
        print(f"Average Reward over {num_episodes} episodes: {average_reward}")
        print(
            f"Average Angle deviation over {num_episodes} episodes: {average_angle_dev}"
        )
        print(
            f"Average lane deviation over {num_episodes} episodes: {average_lane_dev}"
        )