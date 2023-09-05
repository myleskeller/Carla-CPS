#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

import glob
import os
import sys

import cv2
import pytesseract
import keras_ocr
from thefuzz import fuzz
from thefuzz import process


try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (sys.version_info.major, sys.version_info.minor, "win-amd64" if os.name == "nt" else "linux-x86_64")
        )[0]
    )
except IndexError:
    pass

import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, make sure numpy package is installed")


ocr = "tesseract"  # . options: "tesseract", "keras"
intersection_proximity = 30.0  # . in meters; questionable accuracy..
HD_scalar = 2  # . upscaling multiplier of RGB camera data used by OpenCV
sync_by_default = True
opencv_only_near_intersection = True
opencv_enabled = True
hud_default_visibility = False
reached_final_destination = False  # ? maybe delete
player_status = None
yaw_goal = 0

# pipeline = None
pipeline = keras_ocr.pipeline.Pipeline() if ocr == "keras" else None
display_seg = None
HD_surface = None
walker_destination = None


class Intersection:
    def __init__(self, name, center, corners):
        self.name = name
        self.center = center
        self.corners = corners
        # self.visited = []

    def visit(self, location):
        if location in self.corners:
            # self.visited.append(location)
            for corner in self.corners:
                if corner.x == location.x and corner.y == location.y:
                    self.corners.remove(location)
            ## print(len(self.corners))
        else:
            raise NameError("given location not in corners[]")

    def closest(self, location):
        if not self.corners:
            return None  # No corners to compare

        closest_corner = self.corners[0]
        closest_distance = location.distance(closest_corner)

        for corner in self.corners:
            distance = location.distance(corner)
            # print(str(corner) + str(location.distance(corner)))
            if distance < closest_distance:
                closest_corner = corner
                closest_distance = distance

        return closest_corner


"""
# . traffic light locations:
# Transform(Location(x=-119.230034, y=5.093152, z=0.262770), Rotation(pitch=0.000000, yaw=-89.999596, roll=0.000000))
# Transform(Location(x=115.449829, y=35.044945, z=0.222726), Rotation(pitch=0.000000, yaw=90.000114, roll=0.000000))
# Transform(Location(x=-46.171577, y=-73.556618, z=0.254254), Rotation(pitch=0.000000, yaw=-90.000145, roll=0.000000))
# Transform(Location(x=-34.449154, y=-51.020489, z=0.254255), Rotation(pitch=0.000000, yaw=-0.000549, roll=0.000000))
# Transform(Location(x=114.451889, y=21.196003, z=0.254200), Rotation(pitch=0.0map00000, yaw=-0.000031, roll=0.000000))
# Transform(Location(x=89.791885, y=20.922146, z=0.254200), Rotation(pitch=0.000000, yaw=179.999756, roll=0.000000))
# Transform(Location(x=-31.930841, y=20.301954, z=0.254254), Rotation(pitch=0.000000, yaw=-0.000031, roll=0.000000))
# Transform(Location(x=-64.264191, y=7.063309, z=0.254254), Rotation(pitch=0.000000, yaw=-90.000114, roll=0.000000))
# Transform(Location(x=-62.352070, y=20.196909, z=0.254254), Rotation(pitch=0.000000, yaw=179.999786, roll=0.000000))
# Transform(Location(x=-94.928223, y=20.334822, z=0.254256), Rotation(pitch=0.000000, yaw=0.000212, roll=0.000000))
# Transform(Location(x=-59.242249, y=-51.498371, z=0.254254), Rotation(pitch=0.000000, yaw=89.999596, roll=0.000000))
# Transform(Location(x=-119.238182, y=19.004469, z=0.254253), Rotation(pitch=0.000000, yaw=179.999954, roll=0.000000))
# Transform(Location(x=-58.715359, y=145.824570, z=0.254188), Rotation(pitch=0.000000, yaw=179.999954, roll=0.000000))
# Transform(Location(x=-62.099304, y=123.755508, z=0.283086), Rotation(pitch=0.000000, yaw=-90.000114, roll=0.000000))
# Transform(Location(x=-31.636356, y=33.589287, z=0.254254), Rotation(pitch=0.000000, yaw=90.000114, roll=0.000000))
"""

# TODO: figure out what to do about ALL STREETS HAVING THE SAME NAMES.

# . Park Av. & E 79 St. (first intersection)
intersection1_center = carla.Location(x=-106.198769, y=20.588936, z=0.952513)
intersection1A = carla.Location(x=-119.918182, y=39.173973, z=1.103355)
intersection1B = carla.Location(x=-97.409683, y=38.820347, z=1.103900)
intersection1C = carla.Location(x=-93.239342, y=4.090131, z=1.103900)
intersection1D = carla.Location(x=-120.754692, y=3.397407, z=1.101879)
intersection1 = Intersection(
    "1", intersection1_center, [intersection1A, intersection1B, intersection1C, intersection1D]
)

# .second intersection
intersection2_center = carla.Location(x=-45.764511, y=21.156416, z=0.952513)
intersection2A = carla.Location(x=-60.909901, y=36.308926, z=1.101876)
intersection2B = carla.Location(x=-34.275879, y=36.485615, z=1.103901)
intersection2C = carla.Location(x=-35.038628, y=3.013614, z=1.103900)
intersection2D = carla.Location(x=-59.313709, y=2.593773, z=1.103900)
intersection2 = Intersection(
    "2", intersection2_center, [intersection2A, intersection2B, intersection2C, intersection2D]
)

# . third intersection
intersection3_center = carla.Location(x=-46.625027, y=-61.714500, z=0.952513)
intersection3A = carla.Location(x=-62.983162, y=-47.800869, z=1.103900)
intersection3B = carla.Location(x=-31.470959, y=-47.873486, z=1.103900)
intersection3C = carla.Location(x=-31.256779, y=-74.462097, z=1.103900)
intersection3D = carla.Location(x=-62.187439, y=-76.981407, z=1.103900)
intersection3 = Intersection(
    "3", intersection3_center, [intersection3A, intersection3B, intersection3C, intersection3D]
)

# TODO: do this for remaining intersections

# . entrance to parking garage
garageEnterance = carla.Location(x=43.511211, y=-10.272449, z=1.103900)

# . street position of parking garage
garageStreet = carla.Location(x=40.449162, y=-47.947006, z=1.103900)

# . outside storefront starting position
storeStartStreet = carla.Location(x=-132.268295, y=40.859585, z=1.055411)

# storeStartStreet = intersection1_center
# . edge of crosswalk of 1st intersection
# Location(x=-120.932350, y=39.104168, z=1.103317)

cubes = []
cubes.append(intersection1A)
cubes.append(intersection1B)
cubes.append(intersection1C)
cubes.append(intersection1D)
cubes.append(intersection2A)
cubes.append(intersection2B)
cubes.append(intersection2C)
cubes.append(intersection2D)
cubes.append(intersection3A)
cubes.append(intersection3B)
cubes.append(intersection3C)
cubes.append(intersection3D)
cubes.append(garageEnterance)
cubes.append(garageStreet)
cubes.append(storeStartStreet)


def updateSpectatorLocation(spectator, transform, height=200):
    # . positions spectator in top-down view
    # transform.Location.z = height
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=transform.location.x, y=transform.location.y, z=height),
            carla.Rotation(pitch=-90, yaw=-0, roll=-90),
        )
    )


def isGoalReached(player, destination, tolerance, debug=False):
    if debug == True:
        print(player.get_transform().location.distance(destination))
    return player.get_transform().location.distance(destination) < tolerance


def order_points(pts):
    """
    input format:
    [[[482 254]]
    [[483 272]]
    [[492 272]]
    [[492 255]]]

    expected format:
    [(73, 239), (356, 117), (475, 265), (187, 443)]"
    """
    _pts = np.zeros((4, 2), dtype="float32")
    # print("pts", pts)
    # print("type", type(pts))
    # print("size", pts.shape)
    pts.resize((4, 2))
    _pts[0] = (pts[0][0], pts[0][1])
    _pts[1] = (pts[1][0], pts[1][1])
    _pts[2] = (pts[2][0], pts[2][1])
    _pts[3] = (pts[3][0], pts[3][1])
    # print("_pts", _pts)
    pts = _pts

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def get_point_extrema(pts):
    _pts = np.zeros((4, 2), dtype="float32")
    pts.resize((4, 2))
    _pts[0] = (pts[0][0], pts[0][1])
    _pts[1] = (pts[1][0], pts[1][1])
    _pts[2] = (pts[2][0], pts[2][1])
    _pts[3] = (pts[3][0], pts[3][1])
    pts = _pts

    left_extrema = pts[0][0]
    top_extrema = pts[0][1]
    for pt in pts:
        if pt[0] < left_extrema:
            left_extrema = pt[0]
        if pt[1] < top_extrema:
            top_extrema = pt[1]

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    # rect = np.zeros((4, 2), dtype="float32")
    # s = pts.sum(axis=1)

    # # . top-left point will have the smallest sum
    # rect[0] = pts[np.argmin(s)]
    # left_extrema = rect[0][0]
    # top_extrema = rect[0][1]

    # # . bottom-right point will have the largest sum
    # rect[2] = pts[np.argmax(s)]
    # if rect[2][0] < left_extrema:
    #     left_extrema = rect[2][0]
    # if rect[2][1] < top_extrema:
    #     top_extrema = rect[2][1]

    # # now, compute the difference between the points, the
    # diff = np.diff(pts, axis=1)

    # # . top-right point will have the smallest difference
    # rect[1] = pts[np.argmin(diff)]
    # if rect[1][0] < left_extrema:
    #     left_extrema = rect[1][0]
    # if rect[1][1] < top_extrema:
    #     top_extrema = rect[1][1]

    # # . bottom-left will have the largest difference
    # rect[3] = pts[np.argmax(diff)]
    # if rect[3][0] < left_extrema:
    #     left_extrema = rect[3][0]
    # if rect[3][1] < top_extrema:
    #     top_extrema = rect[3][1]

    # return the ordered coordinates
    return (left_extrema, top_extrema)


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def renderCubes(sim_world, cubes):
    for cube in cubes:
        # cube.z += 30.0
        sim_world.debug.draw_point(
            cube,
            size=0.125,
            color=carla.Color(255, 0, 0, 0),
        )


def renderCubeOverPlayer(sim_world, player, fps):
    if fps != 0:
        _fps = fps
    else:
        _fps = 30
    actor_transform = player.get_transform()
    actor_transform.location.z += 30.0

    sim_world.debug.draw_point(
        actor_transform.location,
        size=0.125,
        life_time=math.sqrt((1 / _fps)),
        color=carla.Color(10, 10, 255, 0),
    )


def find_weather_presets():
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    name = lambda x: " ".join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute("generation")) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


class World(object):
    def __init__(self, carla_world, hud, args):
        global sync_by_default
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print("RuntimeError: {}".format(error))
            print("  The server could not send the OpenDRIVE (.xodr) file:")
            print("  Make sure it exists, has the same name of your town, and is correct.")
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.walker_controller = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self.camera_manager_HD = None
        self.seg_camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.restart()  # . recycles restart() function to initialize workd objects
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All,
        ]
        self.traffic_lights = carla_world.get_actors().filter("traffic.traffic_light")  #! list of intersections

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # . Keep same camera config if the camera manager exists
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # . Get a random blueprint
        blueprint = random.choice(get_actor_blueprints(self.world, "walker.pedestrian.*", self._actor_generation))
        blueprint.set_attribute("role_name", self.actor_role_name)
        if blueprint.has_attribute("terramechanics"):
            blueprint.set_attribute("terramechanics", "true")
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
            blueprint.set_attribute("driver_id", driver_id)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "true")

        # . set the max speed
        if blueprint.has_attribute("speed"):
            self.player_max_speed = float(blueprint.get_attribute("speed").recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute("speed").recommended_values[2])
            # self.player_max_speed = 10.589
            # self.player_max_speed_fast = 20.713

        # . Spawn the player
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print("There are no spawn points available in your map/town.")
                print("Please add some Vehicle Spawn Point to your UE4 scene.")
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            global storeStartStreet
            spawn_point = carla.Transform(storeStartStreet, carla.Rotation(pitch=0.000000, yaw=-90, roll=0.000000))
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)

        # . Set up the sensors
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)

        # . initialization of RGB camera
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        ## self.setup_opencv_camera_managers(cam_index, cam_pos_index)

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        walker_controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
        spawn_point = self.player.get_transform()
        self.world.walker_controller = self.world.try_spawn_actor(walker_controller_bp, spawn_point, self.player)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def setup_opencv_camera_managers(self, cam_index, cam_pos_index):
        print("setting up opencv cameras")
        self.seg_camera_manager = CameraManager(self.player, self.hud, self._gamma)  # . segmentation camera
        self.seg_camera_manager.transform_index = cam_pos_index
        self.seg_camera_manager.set_sensor(5, notify=False)

        global HD_scalar
        self.camera_manager_HD = CameraManager(
            self.player, self.hud, self._gamma, HD_scalar=HD_scalar
        )  # . higher resolution camera
        self.camera_manager_HD.transform_index = cam_pos_index
        self.camera_manager_HD.set_sensor(cam_index, notify=False)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification("Weather: %s" % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification("LayerMap selected: %s" % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification("Unloading map layer: %s" % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification("Loading map layer: %s" % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

        self.destroy_opencv_sensors()

    def destroy_opencv_sensors(self):
        print("destroying opencv cameras")
        if self.camera_manager_HD is not None:
            self.camera_manager_HD.sensor.destroy()
        self.camera_manager_HD.sensor = None
        self.camera_manager_HD.index = None
        if self.seg_camera_manager is not None:
            self.seg_camera_manager.sensor.destroy()
        self.seg_camera_manager.sensor = None
        self.seg_camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            # self.camera_manager_HD.sensor,
            # self.seg_camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,
        ]

        if self.camera_manager_HD is not None:
            sensors.append(self.camera_manager_HD.sensor)
        if self.seg_camera_manager is not None:
            sensors.append(self.seg_camera_manager.sensor)

        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                    if opencv_enabled == True:
                        world.camera_manager_HD.toggle_camera()
                        world.seg_camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                    if opencv_enabled == True:
                        world.camera_manager_HD.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                    if opencv_enabled == True:
                        world.camera_manager_HD.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                #! injecting custom keyboard commands --------------------------------
                elif event.key == K_t and pygame.key.get_mods() & KMOD_SHIFT:
                    global ocr
                    if ocr == "tesseract":
                        ocr = "keras"
                    elif ocr == "keras":
                        ocr = "tesseract"
                    print("changed OCR to ", ocr)
                #! -------------------------------------------------------------------
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                    if opencv_enabled == True:
                        world.camera_manager_HD.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.recording_enabled:
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_f:
                        # Toggle ackermann controller
                        self._ackermann_enabled = not self._ackermann_enabled
                        world.hud.show_ackermann_info(self._ackermann_enabled)
                        world.hud.notification(
                            "Ackermann Controller %s" % ("Enabled" if self._ackermann_enabled else "Disabled")
                        )
                    if event.key == K_q:
                        if not self._ackermann_enabled:
                            self._control.gear = 1 if self._control.reverse else -1
                        else:
                            self._ackermann_reverse *= -1
                            # Reset ackermann control
                            self._ackermann_control = carla.VehicleAckermannControl()
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification(
                            "%s Transmission" % ("Manual" if self._control.manual_gear_shift else "Automatic")
                        )
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print(
                                "WARNING: You are currently in asynchronous mode and could "
                                "experience some issues with the traffic simulation"
                            )
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification("Autopilot %s" % ("On" if self._autopilot_enabled else "Off"))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else:  # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else:  # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights:  # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
                # Apply control
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                else:
                    world.player.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    world.hud.update_ackermann_control(self._ackermann_control)

            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.01, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= (
                    min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                )
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world):  # ? maybe ad global rotate left/right here??
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = 0.01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = 0.01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = (
                world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
            )
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = "courier" if os.name == "nt" else "mono"
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == "nt" else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = hud_default_visibility
        self._info_text = []
        self._server_clock = pygame.time.Clock()

        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)

        # .checking if we're near any intersections and dealing with opencv cameras accordingly
        self.opencv_camera_tick(world)

        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = "N" if compass > 270.5 or compass < 89.5 else ""
        heading += "S" if 90.5 < compass < 269.5 else ""
        heading += "E" if 0.5 < compass < 179.5 else ""
        heading += "W" if 180.5 < compass < 359.5 else ""
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter("vehicle.*")
        self._info_text = [
            "Server:  % 16.0f FPS" % self.server_fps,
            "Client:  % 16.0f FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % get_actor_display_name(world.player, truncate=20),
            "Map:     % 20s" % world.map.name.split("/")[-1],
            "Simulation time: % 12s" % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            "Compass:% 17.0f\N{DEGREE SIGN} % 2s" % (compass, heading),
            "Accelero: (%5.1f,%5.1f,%5.1f)" % (world.imu_sensor.accelerometer),
            "Gyroscop: (%5.1f,%5.1f,%5.1f)" % (world.imu_sensor.gyroscope),
            "Location:% 20s" % ("(% 5.1f, % 5.1f)" % (t.location.x, t.location.y)),
            "GNSS:% 24s" % ("(% 2.6f, % 3.6f)" % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            "Height:  % 18.0f m" % t.location.z,
            "",
        ]
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ("Throttle:", c.throttle, 0.0, 1.0),
                ("Steer:", c.steer, -1.0, 1.0),
                ("Brake:", c.brake, 0.0, 1.0),
                ("Reverse:", c.reverse),
                ("Hand brake:", c.hand_brake),
                ("Manual:", c.manual_gear_shift),
                "Gear:        %s" % {-1: "R", 0: "N"}.get(c.gear, c.gear),
            ]
            if self._show_ackermann_info:
                self._info_text += [
                    "",
                    "Ackermann Controller:",
                    "  Target speed: % 8.0f km/h" % (3.6 * self._ackermann_control.speed),
                ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [("Speed:", c.speed, 0.0, 5.556), ("Jump:", c.jump)]
        self._info_text += ["", "Collision:", collision, "", "Number of vehicles: % 8d" % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]
            distance = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2
            )
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append("% 4dm %s" % (d, vehicle_type))

    def opencv_camera_tick(self, world):
        ## global opencv_only_near_intersection
        global player_status
        global opencv_enabled
        if opencv_enabled == True:
            """
            # . checks to see if any traffic lights are nearby
            t = world.player.get_transform()
            d = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2
            )
            traffic_lights = [(d(x.get_location()), x) for x in world.traffic_lights]
            flag = False
            ## distances = []
            for d, traffic_light in sorted(traffic_lights, key=lambda traffic_lights: traffic_lights[0]):
                global intersection_proximity
                if d < intersection_proximity:
                    flag = True
                    ## print(distances)
            """
            # . sets up cameras for opencv (first occurrence)
            if player_status == "rotating" and type(world.camera_manager_HD) == type(None):
                cam_index = world.camera_manager.index if world.camera_manager is not None else 0
                cam_pos_index = world.camera_manager.transform_index if world.camera_manager is not None else 0
                world.setup_opencv_camera_managers(cam_index, cam_pos_index)

            # . destroys opencv cameras
            elif player_status != "rotating" and world.camera_manager_HD is not None:
                if type(world.camera_manager_HD.sensor) != type(None):
                    world.destroy_opencv_sensors()
                    global display_seg
                    global HD_surface
                    display_seg = None
                    HD_surface = None

            # . sets up cameras for opencv (subsequent occurrences)
            elif player_status == "rotating" and type(world.camera_manager_HD.sensor) == type(None):
                cam_index = world.camera_manager.index if world.camera_manager is not None else 0
                cam_pos_index = world.camera_manager.transform_index if world.camera_manager is not None else 0
                world.setup_opencv_camera_managers(cam_index, cam_pos_index)

    def show_ackermann_info(self, enabled):
        self._show_ackermann_info = enabled

    def update_ackermann_control(self, ackermann_control):
        self._ackermann_control = ackermann_control

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text("Error: %s" % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18

        # . check scene with opencv and interpret any text from street name signs
        self.perform_opencv_scan(display)

        self._notifications.render(display)
        self.help.render(display)

    def perform_opencv_scan(self, display):
        global display_seg
        global HD_surface
        if display_seg is not None and HD_surface is not None:
            #! tuneable parameters ----------------------------------------------------------------------------------
            blur_radius = 3  # gaussian blur radius #? (prev. 3)
            kernel_size = 3  # kernel size for dilate/erode #? (prev. 5)
            epsillon_scalar = 0.01  # approximation polygon selection accuracy #? (orig. 0.01, prev. 0.05)
            contour_approx_mthd = cv2.CHAIN_APPROX_SIMPLE  # contour finding method #? (prev: cv2.CONTOURS_MATCH_I3)
            canny_threshold_min = 100  # canny edge detection hysteresis procedure lower threshold #? (prev. 100)
            canny_ratio_min_max = 1  # canny edge detection hysteresis procedure upper threshold scalar #? (prev. 1)
            #! ------------------------------------------------------------------------------------------------------

            try:  # . catches "AttributeError: 'numpy.ndarray' object has no attribute 'get_size'"
                # . converting pygame surface "display_seg" into numpy array for openCV
                _display_seg = pygame.surfarray.array3d(display_seg)  # . convert pygame.surface into 3D numpy array
            except:
                pass
            else:
                _display_seg = _display_seg.swapaxes(0, 1)  # . swaps axes if everything is flipped x->y
                # cv2.imshow("display_seg", _display_seg)

                # . converting pygame surface "display" into numpy array for openCV
                _display = pygame.surfarray.array3d(display)  # . convert pygame.surface into 3D numpy array
                _display = _display.swapaxes(0, 1)  # . swaps axes if everything is flipped x->y
                # cv2.imshow("image", _display)
                _orig = _display.copy()

                # . configuring color filtration
                _display = cv2.cvtColor(_display_seg, cv2.COLOR_BGR2HSV)
                lower_yellow = np.array([90, 100, 43])  # . lower Threshold of yellow in HSV space
                upper_yellow = np.array([100, 255, 255])  # . upper Threshold of yellow in HSV space

                # . adding filtration mask over original image ("_display")
                mask = cv2.inRange(_display, lower_yellow, upper_yellow)
                # cv2.imshow("mask", mask)
                _display = cv2.bitwise_and(_display, _display, mask=mask)
                # cv2.imshow("filtered", _display)

                # . converting image to greyscale
                _display = cv2.cvtColor(_display, cv2.COLOR_BGR2GRAY)

                # . blurring image (important to reduce outline noise)
                # _display = cv2.blur(_display, (blur_radius, blur_radius))  #! parameter
                # cv2.imshow("blurred", _display)

                # . performing edge detection
                # canny_threshold_max = canny_threshold_min * canny_ratio_min_max
                # _display = cv2.Canny(_display, canny_threshold_min, canny_threshold_max, apertureSize=3)
                # cv2.imshow("edges", _display)

                # . enhancing edges of lines
                kernel = np.ones((kernel_size, kernel_size), np.uint8)  #! parameter
                # _display = cv2.dilate(_display, kernel, iterations=2)
                # cv2.imshow("dilated", _display)
                _display = cv2.erode(_display, kernel, iterations=1)
                # cv2.imshow("eroded", _display)

                # . finding contours of detected polygons
                contours = cv2.findContours(_display, cv2.RETR_LIST, contour_approx_mthd)[0]  #! parameter #! parameter
                _display = cv2.drawContours(_orig.copy(), contours, -1, (0, 0, 0), 1)
                # print("Number of contours detected:", len(contours))

                #! alternative attempt using hough lines
                """
                minLineLength = min (mask.shape[0],mask.shape[1])/3
                lines = cv2.HoughLinesP(convexHull_mask, rho = 1,theta = 1*np.pi/180,threshold = 50,
                                        minLineLength = minLineLength,maxLineGap = 50)
                tmp_img = np.zeros((mask.shape[0],mask.shape[1]), dtype = np.uint8)
                for i in range(lines.shape[0]):
                    x1 = lines[i][0][0]
                    y1 = lines[i][0][1]
                    x2 = lines[i][0][2]
                    y2 = lines[i][0][3]
                    cv2.line(tmp_img,(x1,y1),(x2,y2),(255,0,0),2)
                plt.imshow(tmp_img)
                corners = []
                lines = np.squeeze(lines)
                tmp_img = mask.copy()
                if len(lines) == 4:
                    params = []
                    for i in  range(4):
                        params.append(calcParams([lines[i][0], lines[i][1]], [lines[i][2], lines[i][3]]))
                    print (params)
                    for i in range(len(params)):
                        for j in range(i, len(params)):
                            intersec = findIntersection(params[i], params[j])
                            if intersec[1]>0 and intersec[0]>0 and intersec[1]< mask.shape[0] and intersec[0]< mask.shape[1] :
                                print ("Corner: ", intersec)
                                corners.append(intersec)
                    for i in range(4):
                        cv2.circle(tmp_img, corners[i], 5, (255), 5)
                plt.axis('off')
                plt.imshow(tmp_img)
                """

                # . identifying polygons(contours) and applying perspective corrections
                for cnt in contours:
                    x1, y1 = cnt[0][0]
                    approx = cv2.approxPolyDP(cnt, epsillon_scalar * cv2.arcLength(cnt, True), True)
                    if len(approx) == 4:
                        color = (0, 255, 255)
                        _display = cv2.drawContours(_display, [cnt], -1, color, 2)
                        approx_len = str(len(approx)) + "*"
                        cv2.putText(_display, approx_len, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        """
                        # print("rect", cv2.boundingRect(approx))
                        # print("cnt", cnt)
                        # print("approx", approx)
                        # print(approx.shape)
                        """

                        # . perspective correct image data
                        xmin, ymin = get_point_extrema(approx)
                        approx_offset = np.zeros((4, 2), dtype="float32")

                        for i in range(0, 4):
                            approx_offset[i, 0] = approx[i, 0] - xmin
                        for i in range(0, 4):
                            approx_offset[i, 1] = approx[i, 1] - ymin
                        """
                        # print("xmin", xmin, "ymin", ymin)
                        # print("approx after extrema", approx)
                        """
                        global HD_scalar
                        x, y, w, h = cv2.boundingRect(approx * HD_scalar)
                        _display = cv2.polylines(_display, [approx], True, (255, 0, 255), 1)

                        _HD = pygame.surfarray.array3d(HD_surface)  # . convert pygame.surface into 3D numpy array
                        _HD = _HD.swapaxes(0, 1)
                        # x, y, w, h = x * HD_scalar, y * HD_scalar, w * HD_scalar, h * HD_scalar
                        cropped_image = _HD[y : y + h, x : x + w]
                        # cv2.imshow("active contour", cropped_image)
                        warped_image = four_point_transform(cropped_image, approx_offset * HD_scalar)
                        # cv2.imshow("perspective transform", warped_image)
                        x, y, w, h = cv2.boundingRect(approx_offset)
                        aspect_ratio = float(w) / h

                        # . filtering perspective-corrected images
                        #! tuneable parameters ----------------------------------------------------------------------------------
                        aspect_ratio_min = 2  # minimum cutoff aspect ratio of street name signs
                        height_min = 10 * HD_scalar  # minimum cutoff of image data height #? (prev. 10 * HD_scalar)
                        kernel_size = 2  # kernel size for dilate/erode #? (prev. 5)
                        canny_lower = 100  # lower threshold for canny edge detection #? (prev. 100)
                        canny_upper = 200  # upper threshold for canny edge detection #? (prev. 200)
                        thresh_lower = 0  # lower threshold for image binarization #? (prev. 0)
                        thresh_upper = 255  # upper threshold for image binarization #? (prev. 255)
                        #! ------------------------------------------------------------------------------------------------------
                        if aspect_ratio > aspect_ratio_min:
                            if h >= height_min:
                                cv2.imshow("active contour", cropped_image)
                                cv2.imshow("perspective transform", warped_image)
                                _display = cv2.polylines(_display, [approx], True, (255, 0, 255), 3)

                                # . run OCR on image data
                                ocr_image = warped_image.copy()
                                # kernel = np.ones((kernel_size, kernel_size), np.uint8)
                                # ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY)
                                # ocr_image = remove_noise(ocr_image, 5)
                                # ocr_image = deskew(ocr_image) # ERROR: rotation off by 90 degrees
                                # ocr_image = cv2.threshold(ocr_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                                # ocr_image = dilate(ocr_image,kernel_size)
                                # ocr_image = cv2.erode(ocr_image, kernel, iterations=1)
                                # ocr_image = opening(ocr_image, kernel_size)
                                # ocr_image = canny(ocr_image, canny_lower, canny_upper)
                                # . sharpen filter
                                ocr_image = cv2.filter2D(ocr_image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
                                cv2.imshow("pre-processed for OCR", ocr_image)
                                # . 7 == Treat the image as a single text line
                                options = r'--oem 3 --psm 7 -c tessedit_char_whitelist=" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"'
                                result = ""

                                global ocr
                                # . keras-ocr implementation
                                if ocr == "keras" and ocr_image.shape[0] >= 8:  # . keras hates images shorter than 8px
                                    global pipeline
                                    results = pipeline.recognize([ocr_image])[0]
                                    for r, box in results:
                                        result = result + " " + r

                                # . pytesseract implementation
                                elif ocr == "tesseract":
                                    result = pytesseract.image_to_string(ocr_image, config=options)
                                if len(result) > 0:
                                    print(result)
                                street_names = ["E 79 St", "Park Av"]
                                # . removing spaces increases accuracy significantly because spacing is unusually small in street names
                                street_names = [string.replace(" ", "") for string in street_names]
                                # if len(result) > 0:
                                # print(street_names)

                                # . perform fuzzy string comparison on OCR results
                                results_sorted = process.extract(
                                    result.replace(" ", ""), street_names, scorer=fuzz.token_set_ratio
                                )
                                # results_sorted = sorted(results_sorted, key=lambda x: x[1])

                # . adding "opencv active" watermark
                cv2.putText(_display, "OpenCV Active", (1130, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # cv2.imshow("image", image)

                # . adding output to "display" surface to be rendered by pygame
                _display = pygame.image.frombuffer(_display.tobytes(), _display.shape[1::-1], "RGB")
                display.blit(_display, (0, 0))


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split("\n")
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification("Collision with %r" % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.gnss")
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.imu")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)),
        )
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))),
        )
        self.compass = math.degrees(sensor_data.compass)


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5  # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find("sensor.other.radar")
        bp.set_attribute("horizontal_fov", str(35))
        bp.set_attribute("vertical_fov", str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=bound_x + 0.05, z=bound_z + 0.05), carla.Rotation(pitch=5)),
            attach_to=self._parent,
        )
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(pitch=current_rot.pitch + alt, yaw=current_rot.yaw + azi, roll=current_rot.roll),
            ).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(-1.0, 0.0, -1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b),
            )


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, HD_scalar=None):
        self.HD_scalar = HD_scalar
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (
                    carla.Transform(
                        carla.Location(x=-2.0 * bound_x, y=+0.0 * bound_y, z=2.0 * bound_z), carla.Rotation(pitch=8.0)
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(carla.Location(x=+0.8 * bound_x, y=+0.0 * bound_y, z=1.3 * bound_z)),
                    Attachment.Rigid,
                ),
                (
                    carla.Transform(carla.Location(x=+1.9 * bound_x, y=+1.0 * bound_y, z=1.2 * bound_z)),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(x=-2.8 * bound_x, y=+0.0 * bound_y, z=4.6 * bound_z), carla.Rotation(pitch=6.0)
                    ),
                    Attachment.SpringArmGhost,
                ),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0 * bound_y, z=0.4 * bound_z)), Attachment.Rigid),
            ]
        else:
            self._camera_transforms = [
                # (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=0.1, z=1.5)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (
                    carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)),
                    Attachment.SpringArmGhost,
                ),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid),
            ]

        self.transform_index = 1
        self.sensors = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB", {}],
            ["sensor.camera.depth", cc.Raw, "Camera Depth (Raw)", {}],
            ["sensor.camera.depth", cc.Depth, "Camera Depth (Gray Scale)", {}],
            ["sensor.camera.depth", cc.LogarithmicDepth, "Camera Depth (Logarithmic Gray Scale)", {}],
            ["sensor.camera.semantic_segmentation", cc.Raw, "Camera Semantic Segmentation (Raw)", {}],
            [
                "sensor.camera.semantic_segmentation",
                cc.CityScapesPalette,
                "Camera Semantic Segmentation (CityScapes Palette)",
                {},
            ],
            [
                "sensor.camera.instance_segmentation",
                cc.CityScapesPalette,
                "Camera Instance Segmentation (CityScapes Palette)",
                {},
            ],
            ["sensor.camera.instance_segmentation", cc.Raw, "Camera Instance Segmentation (Raw)", {}],
            ["sensor.lidar.ray_cast", None, "Lidar (Ray-Cast)", {"range": "50"}],
            ["sensor.camera.dvs", cc.Raw, "Dynamic Vision Sensor", {}],
            [
                "sensor.camera.rgb",
                cc.Raw,
                "Camera RGB Distorted",
                {
                    "lens_circle_multiplier": "3.0",
                    "lens_circle_falloff": "3.0",
                    "chromatic_aberration_intensity": "0.5",
                    "chromatic_aberration_offset": "0",
                },
            ],
            ["sensor.camera.optical_flow", cc.Raw, "Optical Flow", {}],
            ["sensor.camera.normals", cc.Raw, "Camera Normals", {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                if self.HD_scalar is None:
                    bp.set_attribute("image_size_x", str(hud.dim[0]))
                    bp.set_attribute("image_size_y", str(hud.dim[1]))
                else:
                    bp.set_attribute("image_size_x", str(hud.dim[0] * HD_scalar))
                    bp.set_attribute("image_size_y", str(hud.dim[1] * HD_scalar))
                if bp.has_attribute("gamma"):
                    bp.set_attribute("gamma", str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith("sensor.lidar"):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == "range":
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        # print(index)
        needs_respawn = (
            True if self.index is None else (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        )
        if needs_respawn:
            # print("camera needed respawn")
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1],
            )
            # We need to pass the lambda a weak reference to self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith("sensor.camera.dvs"):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(
                image.raw_data, dtype=np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("pol", np.bool)])
            )
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]["y"], dvs_events[:]["x"], dvs_events[:]["pol"] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith("sensor.camera.optical_flow"):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        elif self.index == 5:
            #! injection point for getting semantic segmentation ------------------------
            image.convert(cc.CityScapesPalette)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            global display_seg
            display_seg = array
            display_seg = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            #! --------------------------------------------------------------------------
        elif self.HD_scalar is not None:
            #! injection point for getting HD RGB camera --------------------------------
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            global HD_surface
            HD_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            #! --------------------------------------------------------------------------
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)


def prepare_scenario(player, spectator):
    global storeStartStreet
    # . positions spectator in top-down view
    # spectator = sim_world.get_spectator()
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=4.635590, y=36.175838, z=199.364319),
            carla.Rotation(pitch=-90, yaw=-0, roll=-90),
        )
    )

    # . positions player
    # for actor in sim_world.get_actors():
    # if actor.attributes.get("role_name") == "hero":
    player.set_transform(
        carla.Transform(
            storeStartStreet,
            carla.Rotation(pitch=0.000000, yaw=-90, roll=0.000000),
        )
    )


def calculate_yaw_to_face(target_object, source_object):
    # Calculate the vector from source_object to target_object
    target_location = target_object
    source_location = source_object
    dx = target_location.x - source_location.x
    dy = target_location.y - source_location.y

    # Calculate the yaw angle
    yaw = math.atan2(dy, dx)

    # Convert radians to degrees
    yaw_degrees = math.degrees(yaw)

    # Adjust the angle to be within the range [0, 360]
    yaw_degrees = (yaw_degrees + 360) % 360
    # compensates for when rotationg other direction is more appropriate
    if yaw_degrees - 360 < abs(yaw_degrees):
        yaw_degrees = yaw_degrees - 360

    return yaw_degrees


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        sim_world = client.get_world()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print(
                "WARNING: You are currently in asynchronous mode and could "
                "experience some issues with the traffic simulation"
            )

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        # print(client.get_available_maps())
        # world = client.load_world("Town01")

        controller = KeyboardControl(world, args.autopilot)

        global cubes
        global player_status
        global yaw_goal

        renderCubes(sim_world, cubes)

        print("starting autopilot")
        sim_world.walker_controller.start()
        sim_world.walker_controller.set_max_speed(speed=5.0)
        destination = intersection1.closest(world.player.get_transform().location)
        sim_world.walker_controller.go_to_location(destination)
        player_status = "walking"
        print(player_status)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()

        # . reposition player here
        prepare_scenario(world.player, sim_world.get_spectator())

        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # print(world.player.get_transform())  #! for debugging

            renderCubeOverPlayer(sim_world, world.player, world.hud.server_fps)
            updateSpectatorLocation(sim_world.get_spectator(), world.player.get_transform(), 200)

            if player_status != "arrived":
                if player_status == "walking":
                    if isGoalReached(world.player, destination, 0.5, debug=False):
                        # . stop walker
                        print("reached waypoint")
                        sim_world.walker_controller.stop()
                        stop = carla.WalkerControl()
                        stop.speed = 0
                        world.player.apply_control(stop)

                        # . mark this corner as visited (in case we can't see shit during OCR)
                        intersection1.visit(intersection1.closest(world.player.get_transform().location))

                        player_status = "rotating"
                        print(player_status)

                        # . rotate towards center of intersection until 90 degree sweep accomplished
                        # ? check for the sign of these values in case they are not negative?
                        yaw = round(calculate_yaw_to_face(intersection1.center, world.player.get_transform().location))
                        print("yaw_to_face: " + str(yaw))
                        yaw_goal = round(yaw - (90 + yaw + world.player.get_transform().rotation.yaw))
                        print("yaw_goal: " + str(yaw_goal))

                elif player_status == "rotating":
                    current_pos = world.player.get_transform()
                    current_pos.rotation.yaw -= 0.5  # ? check for the sign of this value in case it is not negative?
                    world.player.set_transform(current_pos)

                    # print(str(yaw_goal) + " != " + str(round(world.player.get_transform().rotation.yaw)))
                    if round(world.player.get_transform().rotation.yaw) == yaw_goal:
                        stop = carla.WalkerControl()
                        stop.speed = 0
                        world.player.apply_control(stop)
                        player_status = "computing"
                        print(player_status)
                        # TODO: add logic to compute next intersection based off of OCR output
                        street_names_read = False  #! temporary
                        print("not enough OCR data; moving to next intersection corner")

                        if street_names_read == False:
                            if len(intersection1.corners) > 0:
                                destination = intersection1.closest(world.player.get_transform().location)
                                sim_world.walker_controller.start()
                                sim_world.walker_controller.go_to_location(destination)
                                player_status = "walking"
                                print(player_status)
                            else:
                                # TODO: add logic to navigate to random adjacent intersection out of frustration
                                player_status = "guessing"
                                print(player_status)

    finally:
        """
        # print("All landmarks in world:")
        # map = sim_world.get_map()
        # waypoint = map.get_waypoint(world.player.get_location())
        # landmarks = waypoint.get_landmarks(2000.0, False)
        # for landmark in landmarks:
        #     print(
        #         "name:",
        #         landmark.name,
        #         "\ttype:",
        #         landmark.type,
        #         "\tsub_type:",
        #         landmark.sub_type,
        #         "\tvalue:",
        #         landmark.value,
        #         "\ttext:",
        #         landmark.text,
        #     )
        """

        if original_settings:
            sim_world.apply_settings(original_settings)

        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    argparser.add_argument("-v", "--verbose", action="store_true", dest="debug", help="print debug information")
    argparser.add_argument(
        "--host", metavar="H", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)"
    )
    argparser.add_argument(
        "-p", "--port", metavar="P", default=2000, type=int, help="TCP port to listen to (default: 2000)"
    )
    argparser.add_argument("-a", "--autopilot", action="store_true", help="enable autopilot")
    argparser.add_argument(
        "--res", metavar="WIDTHxHEIGHT", default="1280x720", help="window resolution (default: 1280x720)"
    )
    ## argparser.add_argument("--res", metavar="WIDTHxHEIGHT", default="1920x1080", help="window resolution (default: 1280x720)")
    argparser.add_argument(
        "--filter", metavar="PATTERN", default="vehicle.*", help='actor filter (default: "vehicle.*")'
    )
    argparser.add_argument(
        "--generation",
        metavar="G",
        default="2",
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")',
    )
    argparser.add_argument("--rolename", metavar="NAME", default="hero", help='actor role name (default: "hero")')
    argparser.add_argument("--gamma", default=2.2, type=float, help="Gamma correction of the camera (default: 2.2)")
    # argparser.add_argument("--sync", default=True, action="store_true", help="Activate synchronous mode execution")
    argparser.add_argument("--sync", action="store_true", help="Activate synchronous mode execution")
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("listening to server %s:%s", args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")


if __name__ == "__main__":
    main()
