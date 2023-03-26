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


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
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

import time

number_of_walkers = 50
number_of_vehicles = 5

vehicle_obstacle_distance = 10
walker_obstacle_distance = 5
node_obstacle_distance = 50  # . for now assuming that these include powered infrastructure

default_message = "No volcanoes yet."
hero_message = "THERE IS A VOLCANO."

start_time = 0
fps = 0

obstacle_sensors = []  # . only here to keep carla from (incorrectly) whining about sensors going out of scope
AWayNet_list = None  # . holds data structures for each AWayNet-compatible actor


class AWayNet:
    def __init__(self, message, timestamp, id):
        global start_time
        self.id = id
        self.message = message
        self.timestamp = timestamp
        self.last_update_time = time.time() - start_time
        self.last_update_actor = None
        self.last_near = None

    def setMessage(self, message, timestamp):
        self.message = message
        self.timestamp = timestamp

    def getMessage(self):
        return self.message

    def getTimestamp(self):
        return self.timestamp

    def setLastUpdateInfo(self, actor):
        global start_time
        self.last_update_time = time.time() - start_time
        self.last_update_actor = actor

    def isNewerThan(self, other_actor):
        if self.timestamp > other_actor.getTimestamp():
            other_actor.setLastUpdateInfo(self.id)
        return self.timestamp > other_actor.getTimestamp()

    def print(self):
        return str("msg:'" + self.message + "', timestamp:" + str(self.timestamp))


class AWayNetList:
    global default_message
    global hero_message
    global start_time

    dictionary = dict()
    hero_id = 0
    first_contact = False
    final_contact = False
    contacted_actors = 0

    def getHeroID(self, actor_list=None):
        if actor_list == None:
            return self.hero_id
        else:
            for actor in actor_list:
                if actor.attributes.get("role_name") == "hero":
                    self.hero_id = actor.id
                    print("the hero is ID " + str(actor.id) + ".")
                    return actor.id

    def containsActor(self, actor_id):
        return actor_id in self.dictionary.keys()

    def getAllParticipantIDs(self):
        return list(self.dictionary.keys())

    def updateOlderTimestamp(self, A, B):
        global start_time
        if self.dictionary[A].isNewerThan(self.dictionary[B]):
            self.dictionary[B].setMessage(
                self.dictionary[A].getMessage(),
                self.dictionary[A].getTimestamp(),
            )
            print(
                str(B)
                + ' updated to "'
                + self.dictionary[B].getMessage()
                + '" by '
                + str(A)
                + " @"
                + str(round(time.time() - start_time))
                + "s"
            )

            AWayNet_list.checkPropagation()

        elif self.dictionary[B].isNewerThan(self.dictionary[A]):
            # . if it is out of date, assign it the message from detector_actor to other_actor
            self.dictionary[A].setMessage(
                self.dictionary[B].getMessage(),
                self.dictionary[B].getTimestamp(),
            )

            print(
                str(A)
                + ' updated to "'
                + self.dictionary[A].getMessage()
                + '" by '
                + str(B)
                + " @"
                + str(round(time.time() - start_time))
                + "s"
            )

            AWayNet_list.checkPropagation()

    def timestampsAreDifferent(self, detector_actor, other_actor):
        print("\n" + str(detector_actor) + "'s message is newer than " + str(other_actor) + "'s")
        return self.dictionary[detector_actor].getTimestamp() != self.dictionary[other_actor].getTimestamp()

    def checkPropagation(self):
        global start_time
        # . most of this will become obsolete as the AWAyNet message data structure evolves
        if self.first_contact == False or self.final_contact == False:
            contacted_actors = 0
            for actor in self.dictionary:
                if self.dictionary[actor].getMessage() == hero_message:
                    contacted_actors = contacted_actors + 1

            if self.contacted_actors != contacted_actors:
                print(str(contacted_actors) + "/" + str(len(self.dictionary)))
                self.contacted_actors = contacted_actors

            if contacted_actors == 2 and self.first_contact == False:
                print("#================================================================")
                print("first contact made.")
                print("#================================================================")
                self.first_contact = True
                # . start timer
                start_time = time.time()

            elif contacted_actors == len(self.dictionary) and self.final_contact == False:
                self.final_contact = True
                # . stop timer
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("#================================================================")
                print("final contact made.")
                print("Total propagation time: " + str(round(elapsed_time)))
                print("#================================================================")

    def print(self):
        for key, value in self.dictionary.items():
            print(str(key) + ": " + value.print())

    def __init__(self, world):
        # . find all actors in world
        all_vehicle_actors = world.get_actors().filter("vehicle.*")
        all_walker_actors = world.get_actors().filter("walker.pedestrian.*")
        # TODO update to include semantic nodes
        all_nodes = world.get_actors().filter("???")
        # TODO store state from carla.TrafficLight.get_state() in AWayNet data structure
        # carla.TrafficLightState()

        self.getHeroID(all_vehicle_actors)

        # . finds all vehicles and creates accompanying AWayNet data structure
        for actor in all_vehicle_actors:
            # . instantiate AWayNet data structure with default values and modify so hero timestamp is newer
            timestamp = world.get_snapshot().timestamp.elapsed_seconds - 500.0
            self.dictionary[actor.id] = AWayNet(default_message, timestamp, actor.id)

            # . create obstacle detection sensor and attach sensor to vehicle (parent actor)
            obstacle_sensor = AObstacleDetectionSensor(actor, vehicle_obstacle_distance)
            obstacle_sensors.append(obstacle_sensor)

        # . finds all walkers and creates accompanying AWayNet data structure
        for actor in all_walker_actors:
            # . instantiate AWayNet data structure with default values and modify so hero timestamp is newer
            timestamp = world.get_snapshot().timestamp.elapsed_seconds - 500.0
            self.dictionary[actor.id] = AWayNet(default_message, timestamp, actor.id)

            # . create obstacle detection sensor and attach sensor to walker (parent actor)
            obstacle_sensor = AObstacleDetectionSensor(actor, walker_obstacle_distance)
            obstacle_sensors.append(obstacle_sensor)

        # . finds all walkers and creates accompanying AWayNet data structure
        for actor in all_nodes:
            # . instantiate AWayNet data structure with default values and modify so hero timestamp is newer
            timestamp = world.get_snapshot().timestamp.elapsed_seconds - 500.0
            self.dictionary[actor.id] = AWayNet(default_message, timestamp, actor.id)

            # . create obstacle detection sensor and attach sensor to nodes (parent actor)
            obstacle_sensor = AObstacleDetectionSensor(actor, node_obstacle_distance)
            obstacle_sensors.append(obstacle_sensor)

        # . we give the hero a unique message and timestamp so that the message proapgates
        hero_timestamp = world.get_snapshot().timestamp.elapsed_seconds
        self.dictionary[self.hero_id] = AWayNet(hero_message, hero_timestamp, actor.id)

    def clear(self):
        self.dictionary.clear()
        self.hero_id = 0
        self.first_contact = False
        self.final_contact = False
        self.contacted_actors = 0


def renderCubeOverPropagatedActors(world, originActor, actorIDList):
    # . this whole function will change depending on how many variants of the message there are
    global fps
    global AWayNet_list

    if fps != 0:
        _fps = fps
    else:
        _fps = 30

    actors = world.get_actors()
    for actor in actors:
        if AWayNet_list.containsActor(actor.id):
            # . selects actors that have NOT received new message
            if actorIDList[originActor].isNewerThan(actorIDList[actor.id]):
                actor_transform = actor.get_transform()
                actor_transform.location.z += 3.0

                world.debug.draw_point(
                    actor_transform.location,
                    size=0.125,
                    life_time=math.sqrt((1 / _fps)),
                    color=carla.Color(255, 0, 0, 0),
                )
            # . selects actors that have received new message
            else:
                actor_transform = actor.get_transform()
                actor_transform.location.z += 3.0

                # . draws cube above awaynet participants to indicate update status
                world.debug.draw_point(
                    actor_transform.location,
                    size=0.125,
                    life_time=math.sqrt((1 / _fps)),
                    color=carla.Color(0, 255, 0, 0),
                )


def spawn_vehicles(args, client, sim_world, traffic_manager, vehicles_list):
    global number_of_vehicles

    blueprints = get_actor_blueprints(sim_world, "vehicle.*", "All")
    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = sim_world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif number_of_vehicles > number_of_spawn_points:
        msg = "requested %d vehicles, but could only find %d spawn points"
        logging.warning(msg, number_of_vehicles, number_of_spawn_points)
        number_of_vehicles = number_of_spawn_points

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        blueprint = random.choice(blueprints)
        while blueprint.has_attribute("has_lights") == False:
            blueprint = random.choice(blueprints)

        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
            blueprint.set_attribute("driver_id", driver_id)
        else:
            blueprint.set_attribute("role_name", "autopilot")

            # . spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, args.sync):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    print("Spawned %d vehicles." % (len(vehicles_list)))

    return vehicles_list


def spawn_walkers(args, client, sim_world, walkers_list, walkers_ids):
    SpawnActor = carla.command.SpawnActor
    global number_of_walkers

    blueprintsWalkers = get_actor_blueprints(sim_world, "walker.pedestrian.*", "2")

    percentagePedestriansRunning = 0.0  # how many pedestrians will run
    percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road

    spawn_points = []
    for i in range(number_of_walkers):
        spawn_point = carla.Transform()
        loc = sim_world.get_random_location_from_navigation()
        if loc != None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)
        # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
            # set the max speed
        if walker_bp.has_attribute("speed"):
            if random.random() > percentagePedestriansRunning:
                # walking
                walker_speed.append(walker_bp.get_attribute("speed").recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute("speed").recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = sim_world.get_blueprint_library().find("controller.ai.walker")
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        walkers_ids.append(walkers_list[i]["con"])
        walkers_ids.append(walkers_list[i]["id"])
    all_actors = sim_world.get_actors(walkers_ids)

    if args.sync:
        sim_world.tick()
    else:
        sim_world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
    sim_world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(walkers_ids), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(sim_world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))
    print("Spawned %d walkers." % (len(walkers_list)))
    return walkers_list, walkers_ids


def prepare_scenario(sim_world):
    # . sets weather
    sim_world.set_weather(carla.WeatherParameters.ClearSunset)
    # sim_world.set_weather(carla.WeatherParameters.ClearNight)
    # sim_world.set_weather(carla.WeatherParameters.ClearNoon)

    # . positions spectator in top-down view
    spectator = sim_world.get_spectator()
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=4.635590, y=36.175838, z=192.364319),
            carla.Rotation(pitch=-90, yaw=-0, roll=-90),
        )
    )

    # . positions player (ego) out of the way so we control introduction time of new message
    for actor in sim_world.get_actors():
        if actor.attributes.get("role_name") == "hero":
            actor.set_transform(
                carla.Transform(
                    carla.Location(x=136.513748, y=95.039558, z=0.090692),
                    carla.Rotation(pitch=0.006557, yaw=-179.795471, roll=-0.041992),
                )
            )


#! ==============================================================================


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
        self.collision_sensor = None
        self.obstacle_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.restart()
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

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))
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
        # set the max speed
        if blueprint.has_attribute("speed"):
            self.player_max_speed = float(blueprint.get_attribute("speed").recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute("speed").recommended_values[2])

        # Spawn the player.
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
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.obstacle_sensor = AObstacleDetectionSensor(self.player, vehicle_obstacle_distance)  ##, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

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
        # print(self.player.get_transform())

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.obstacle_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,
        ]
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
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
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
                    min(
                        abs(self._ackermann_control.speed),
                        round(milliseconds * 0.005, 2),
                    )
                    * self._ackermann_reverse
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

    def _parse_walker_keys(self, keys, milliseconds, world):
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
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        global fps
        self.server_fps = fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        # print(t)
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
        self._info_text += [
            "",
            "Collision:",
            collision,
            "",
            "Number of vehicles: % 8d" % len(vehicles),
        ]
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
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8),
                                (6, 6),
                            )
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


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


class AObstacleDetectionSensor(object):
    def __init__(self, parent_actor, hit_radius):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self.world = self._parent.get_world()
        self.debug = world.debug

        bp = world.get_blueprint_library().find("sensor.other.obstacle")

        bp.set_attribute("only_dynamics", str(True))  # . only applies to 'dynamic' objects
        bp.set_attribute("debug_linetrace", str(False))  #! ☠️☠️☠️ setting to 'True' murders fps
        bp.set_attribute("distance", str(1))  # . '1' should convert capsule to sphere
        bp.set_attribute("hit_radius", str(hit_radius))  # ? in meters?

        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # . We need to pass the lambda a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: AObstacleDetectionSensor._on_obstacle(weak_self, event))

    @staticmethod
    def _on_obstacle(weak_self, event):
        global AWayNet_list
        self = weak_self()
        if not self:
            return

        # other_actor = get_actor_display_name(event.other_actor)  #. Actor detected as an obstacle.
        # actor = get_actor_display_name(event.actor) #. Actor that detected the obstacle (will be the obstacle sensor, not the parent).
        # distance = event.distance #. Distance from actor to other_actor.

        detector_actor = event.actor.parent.id
        other_actor = event.other_actor.id

        if AWayNet_list is not None and AWayNet_list.containsActor(other_actor):
            # . check to see if other_actor's info is outdated
            AWayNet_list.updateOlderTimestamp(detector_actor, other_actor)


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
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z + 0.05),
                carla.Rotation(pitch=5),
            ),
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
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll,
                ),
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
    def __init__(self, parent_actor, hud, gamma_correction):
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
                        carla.Location(x=-2.0 * bound_x, y=+0.0 * bound_y, z=2.0 * bound_z),
                        carla.Rotation(pitch=8.0),
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
                        carla.Location(x=-2.8 * bound_x, y=+0.0 * bound_y, z=4.6 * bound_z),
                        carla.Rotation(pitch=6.0),
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(carla.Location(x=-1.0, y=-1.0 * bound_y, z=0.4 * bound_z)),
                    Attachment.Rigid,
                ),
            ]
        else:
            self._camera_transforms = [
                (
                    carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)),
                    Attachment.SpringArmGhost,
                ),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (
                    carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)),
                    Attachment.Rigid,
                ),
            ]

        self.transform_index = 1
        self.sensors = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB", {}],
            ["sensor.camera.depth", cc.Raw, "Camera Depth (Raw)", {}],
            ["sensor.camera.depth", cc.Depth, "Camera Depth (Gray Scale)", {}],
            [
                "sensor.camera.depth",
                cc.LogarithmicDepth,
                "Camera Depth (Logarithmic Gray Scale)",
                {},
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.Raw,
                "Camera Semantic Segmentation (Raw)",
                {},
            ],
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
            [
                "sensor.camera.instance_segmentation",
                cc.Raw,
                "Camera Instance Segmentation (Raw)",
                {},
            ],
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
                bp.set_attribute("image_size_x", str(hud.dim[0]))
                bp.set_attribute("image_size_y", str(hud.dim[1]))
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
        needs_respawn = (
            True if self.index is None else (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        )
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1],
            )
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
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
                image.raw_data,
                dtype=np.dtype(
                    [
                        ("x", np.uint16),
                        ("y", np.uint16),
                        ("t", np.int64),
                        ("pol", np.bool),
                    ]
                ),
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
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)


def game_loop(args):
    global AWayNet_list
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
        else:
            print("[!] This script was executed in Asynchronous mode. Was this intentional?")

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You're currently in asynchronous mode and could experience issues with traffic simulation")

        # . vehicle generation
        vehicles_list = []
        vehicles_list = spawn_vehicles(args, client, sim_world, traffic_manager, vehicles_list)

        # . walker generation
        walkers_list = []
        walkers_ids = []
        walkers_list, walkers_ids = spawn_walkers(args, client, sim_world, walkers_list, walkers_ids)

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()

        # . instantiate awaynet entity list
        AWayNet_list = AWayNetList(sim_world)

        # . set up environment for our particular test scenario (weather, spectator position, etc.)
        prepare_scenario(sim_world)

        while True:
            if args.sync:
                sim_world.tick()
                renderCubeOverPropagatedActors(sim_world, AWayNet_list.getHeroID(), AWayNet_list.dictionary)
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if original_settings:
            sim_world.apply_settings(original_settings)

        print("Destroying %d vehicles" % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        print("Destroying %d walkers" % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in walkers_ids])

        # . the scorched earth method to remove actors when i implement something incorrectly
        """
        all_vehicle_actors = sim_world.get_actors().filter("vehicle.*")
        client.apply_batch([carla.command.DestroyActor(x.id) for x in all_vehicle_actors])
        all_walker_actors = sim_world.get_actors().filter("walker.pedestrian.*")
        client.apply_batch([carla.command.DestroyActor(x.id) for x in all_walker_actors])
        """

        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        # . holds data structures for each AWayNet-compatible actor
        AWayNet_list.clear()

        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="print debug information",
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument("-a", "--autopilot", action="store_true", help="enable autopilot")
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="1280x720",
        help="window resolution (default: 1280x720)",
    )
    argparser.add_argument(
        "--filter",
        metavar="PATTERN",
        default="vehicle.*",
        help='actor filter (default: "vehicle.*")',
    )
    argparser.add_argument(
        "--generation",
        metavar="G",
        default="2",
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")',
    )
    argparser.add_argument(
        "--rolename",
        metavar="NAME",
        default="hero",
        help='actor role name (default: "hero")',
    )
    argparser.add_argument(
        "--gamma",
        default=2.2,
        type=float,
        help="Gamma correction of the camera (default: 2.2)",
    )
    argparser.add_argument("--sync", action="store_true", help="Activate synchronous mode execution")
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("listening to server %s:%s", args.host, args.port)

    # print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")


if __name__ == "__main__":
    main()
