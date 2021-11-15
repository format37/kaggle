import numpy as np
import math

### Game
import math
from typing import List, Dict

import json

GAME_CONSTANTS = json.loads("""{
  "UNIT_TYPES": {
    "WORKER": 0,
    "CART": 1
  },
  "RESOURCE_TYPES": {
    "WOOD": "wood",
    "COAL": "coal",
    "URANIUM": "uranium"
  },
  "DIRECTIONS": {
    "NORTH": "n",
    "WEST": "w",
    "EAST": "e",
    "SOUTH": "s",
    "CENTER": "c"
  },
  "PARAMETERS": {
    "DAY_LENGTH": 30,
    "NIGHT_LENGTH": 10,
    "MAX_DAYS": 360,
    "LIGHT_UPKEEP": {
      "CITY": 23,
      "WORKER": 4,
      "CART": 10
    },
    "WOOD_GROWTH_RATE": 1.025,
    "MAX_WOOD_AMOUNT": 500,
    "CITY_BUILD_COST": 100,
    "CITY_ADJACENCY_BONUS": 5,
    "RESOURCE_CAPACITY": {
      "WORKER": 100,
      "CART": 2000
    },
    "WORKER_COLLECTION_RATE": {
      "WOOD": 20,
      "COAL": 5,
      "URANIUM": 2
    },
    "RESOURCE_TO_FUEL_RATE": {
      "WOOD": 1,
      "COAL": 10,
      "URANIUM": 40
    },
    "RESEARCH_REQUIREMENTS": {
      "COAL": 50,
      "URANIUM": 200
    },
    "CITY_ACTION_COOLDOWN": 10,
    "UNIT_ACTION_COOLDOWN": {
      "CART": 3,
      "WORKER": 2
    },
    "MAX_ROAD": 6,
    "MIN_ROAD": 0,
    "CART_ROAD_DEVELOPMENT_RATE": 0.75,
    "PILLAGE_RATE": 0.5
  }
}""")

class Constants:
    class INPUT_CONSTANTS:
        RESEARCH_POINTS = "rp"
        RESOURCES = "r"
        UNITS = "u"
        CITY = "c"
        CITY_TILES = "ct"
        ROADS = "ccd"
        DONE = "D_DONE"
    class DIRECTIONS:
        NORTH = "n"
        WEST = "w"
        SOUTH = "s"
        EAST = "e"
        CENTER = "c"
    class UNIT_TYPES:
        WORKER = 0
        CART = 1
    class RESOURCE_TYPES:
        WOOD = "wood"
        URANIUM = "uranium"
        COAL = "coal"

DIRECTIONS = Constants.DIRECTIONS
RESOURCE_TYPES = Constants.RESOURCE_TYPES

UNIT_TYPES = Constants.UNIT_TYPES


class Player:
    def __init__(self, team):
        self.team = team
        self.research_points = 0
        self.units: list[Unit] = []
        self.cities: Dict[str, City] = {}
        self.city_tile_count = 0
    def researched_coal(self) -> bool:
        return self.research_points >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"]
    def researched_uranium(self) -> bool:
        return self.research_points >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]


class City:
    def __init__(self, teamid, cityid, fuel, light_upkeep):
        self.cityid = cityid
        self.team = teamid
        self.fuel = fuel
        self.citytiles: list[CityTile] = []
        self.light_upkeep = light_upkeep
    def _add_city_tile(self, x, y, cooldown):
        ct = CityTile(self.team, self.cityid, x, y, cooldown)
        self.citytiles.append(ct)
        return ct
    def get_light_upkeep(self):
        return self.light_upkeep


class CityTile:
    def __init__(self, teamid, cityid, x, y, cooldown):
        self.cityid = cityid
        self.team = teamid
        self.pos = Position(x, y)
        self.cooldown = cooldown
    def can_act(self) -> bool:
        """
        Whether or not this unit can research or build
        """
        return self.cooldown < 1
    def research(self) -> str:
        """
        returns command to ask this tile to research this turn
        """
        return "r {} {}".format(self.pos.x, self.pos.y)
    def build_worker(self) -> str:
        """
        returns command to ask this tile to build a worker this turn
        """
        return "bw {} {}".format(self.pos.x, self.pos.y)
    def build_cart(self) -> str:
        """
        returns command to ask this tile to build a cart this turn
        """
        return "bc {} {}".format(self.pos.x, self.pos.y)


class Cargo:
    def __init__(self):
        self.wood = 0
        self.coal = 0
        self.uranium = 0

    def __str__(self) -> str:
        return f"Cargo | Wood: {self.wood}, Coal: {self.coal}, Uranium: {self.uranium}"


class Unit:
    def __init__(self, teamid, u_type, unitid, x, y, cooldown, wood, coal, uranium):
        self.pos = Position(x, y)
        self.team = teamid
        self.id = unitid
        self.type = u_type
        self.cooldown = cooldown
        self.cargo = Cargo()
        self.cargo.wood = wood
        self.cargo.coal = coal
        self.cargo.uranium = uranium
    def is_worker(self) -> bool:
        return self.type == UNIT_TYPES.WORKER

    def is_cart(self) -> bool:
        return self.type == UNIT_TYPES.CART

    def get_cargo_space_left(self):
        """
        get cargo space left in this unit
        """
        spaceused = self.cargo.wood + self.cargo.coal + self.cargo.uranium
        if self.type == UNIT_TYPES.WORKER:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] - spaceused
        else:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] - spaceused
    
    def can_build(self, game_map) -> bool:
        """
        whether or not the unit can build where it is right now
        """
        cell = game_map.get_cell_by_pos(self.pos)
        if not cell.has_resource() and self.can_act() and (self.cargo.wood + self.cargo.coal + self.cargo.uranium) >= GAME_CONSTANTS["PARAMETERS"]["CITY_BUILD_COST"]:
            return True
        return False

    def can_act(self) -> bool:
        """
        whether or not the unit can move or not. This does not check for potential collisions into other units or enemy cities
        """
        return self.cooldown < 1

    def move(self, dir) -> str:
        """
        return the command to move unit in the given direction
        """
        return "m {} {}".format(self.id, dir)

    def transfer(self, dest_id, resourceType, amount) -> str:
        """
        return the command to transfer a resource from a source unit to a destination unit as specified by their ids
        """
        return "t {} {} {} {}".format(self.id, dest_id, resourceType, amount)

    def build_city(self) -> str:
        """
        return the command to build a city right under the worker
        """
        return "bcity {}".format(self.id)

    def pillage(self) -> str:
        """
        return the command to pillage whatever is underneath the worker
        """
        return "p {}".format(self.id)

class Resource:
    def __init__(self, r_type: str, amount: int):
        self.type = r_type
        self.amount = amount


class Cell:
    def __init__(self, x, y):
        self.pos = Position(x, y)
        self.resource: Resource = None
        self.citytile = None
        self.road = 0
    def has_resource(self):
        return self.resource is not None and self.resource.amount > 0


class GameMap:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.map: List[List[Cell]] = [None] * height
        for y in range(0, self.height):
            self.map[y] = [None] * width
            for x in range(0, self.width):
                self.map[y][x] = Cell(x, y)

    def get_cell_by_pos(self, pos) -> Cell:
        return self.map[pos.y][pos.x]

    def get_cell(self, x, y) -> Cell:
        return self.map[y][x]

    def _setResource(self, r_type, x, y, amount):
        """
        do not use this function, this is for internal tracking of state
        """
        cell = self.get_cell(x, y)
        cell.resource = Resource(r_type, amount)


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, pos) -> int:
        return abs(pos.x - self.x) + abs(pos.y - self.y)

    def distance_to(self, pos):
        """
        Returns Manhattan (L1/grid) distance to pos
        """
        return self - pos

    def is_adjacent(self, pos):
        return (self - pos) <= 1

    def __eq__(self, pos) -> bool:
        return self.x == pos.x and self.y == pos.y

    def equals(self, pos):
        return self == pos

    def translate(self, direction, units) -> 'Position':
        if direction == DIRECTIONS.NORTH:
            return Position(self.x, self.y - units)
        elif direction == DIRECTIONS.EAST:
            return Position(self.x + units, self.y)
        elif direction == DIRECTIONS.SOUTH:
            return Position(self.x, self.y + units)
        elif direction == DIRECTIONS.WEST:
            return Position(self.x - units, self.y)
        elif direction == DIRECTIONS.CENTER:
            return Position(self.x, self.y)

    def direction_to(self, target_pos: 'Position') -> DIRECTIONS:
        """
        Return closest position to target_pos from this position
        """
        check_dirs = [
            DIRECTIONS.NORTH,
            DIRECTIONS.EAST,
            DIRECTIONS.SOUTH,
            DIRECTIONS.WEST,
        ]
        closest_dist = self.distance_to(target_pos)
        closest_dir = DIRECTIONS.CENTER
        for direction in check_dirs:
            newpos = self.translate(direction, 1)
            dist = target_pos.distance_to(newpos)
            if dist < closest_dist:
                closest_dir = direction
                closest_dist = dist
        return closest_dir

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


INPUT_CONSTANTS = Constants.INPUT_CONSTANTS

class Game:
    def _initialize(self, messages):
        """
        initialize state
        """
        self.id = int(messages[0])
        self.turn = -1
        # get some other necessary initial input
        mapInfo = messages[1].split(" ")
        self.map_width = int(mapInfo[0])
        self.map_height = int(mapInfo[1])
        self.map = GameMap(self.map_width, self.map_height)
        self.players = [Player(0), Player(1)]

    def _end_turn(self):
        print("D_FINISH")

    def _reset_player_states(self):
        self.players[0].units = []
        self.players[0].cities = {}
        self.players[0].city_tile_count = 0
        self.players[1].units = []
        self.players[1].cities = {}
        self.players[1].city_tile_count = 0

    def _update(self, messages):
        """
        update state
        """
        self.map = GameMap(self.map_width, self.map_height)
        self.turn += 1
        self._reset_player_states()

        for update in messages:
            if update == "D_DONE":
                break
            strs = update.split(" ")
            input_identifier = strs[0]
            if input_identifier == INPUT_CONSTANTS.RESEARCH_POINTS:
                team = int(strs[1])
                self.players[team].research_points = int(strs[2])
            elif input_identifier == INPUT_CONSTANTS.RESOURCES:
                r_type = strs[1]
                x = int(strs[2])
                y = int(strs[3])
                amt = int(float(strs[4]))
                self.map._setResource(r_type, x, y, amt)
            elif input_identifier == INPUT_CONSTANTS.UNITS:
                unittype = int(strs[1])
                team = int(strs[2])
                unitid = strs[3]
                x = int(strs[4])
                y = int(strs[5])
                cooldown = float(strs[6])
                wood = int(strs[7])
                coal = int(strs[8])
                uranium = int(strs[9])
                self.players[team].units.append(Unit(team, unittype, unitid, x, y, cooldown, wood, coal, uranium))
            elif input_identifier == INPUT_CONSTANTS.CITY:
                team = int(strs[1])
                cityid = strs[2]
                fuel = float(strs[3])
                lightupkeep = float(strs[4])
                self.players[team].cities[cityid] = City(team, cityid, fuel, lightupkeep)
            elif input_identifier == INPUT_CONSTANTS.CITY_TILES:
                team = int(strs[1])
                cityid = strs[2]
                x = int(strs[3])
                y = int(strs[4])
                cooldown = float(strs[5])
                city = self.players[team].cities[cityid]
                citytile = city._add_city_tile(x, y, cooldown)
                self.map.get_cell(x, y).citytile = citytile
                self.players[team].city_tile_count += 1;
            elif input_identifier == INPUT_CONSTANTS.ROADS:
                x = int(strs[1])
                y = int(strs[2])
                road = float(strs[3])
                self.map.get_cell(x, y).road = road

###########################
### Define helper functions

# this snippet finds all resources stored on the map and puts them into a list so we can search over them
def find_resources(game_state):
    resource_tiles: list[Cell] = []
    width, height = game_state.map_width, game_state.map_height
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
    return resource_tiles


# the next snippet finds the closest resources that we can mine given position on a map
def find_closest_resources(pos, player, resource_tiles):
    closest_dist = math.inf
    closest_resource_tile = None
    for resource_tile in resource_tiles:
        # we skip over resources that we can't mine due to not having researched them
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile


def find_closest_city_tile(pos, player):
    closest_city_tile = None
    if len(player.cities) > 0:
        closest_dist = math.inf
        # the cities are stored as a dictionary mapping city id to the city object, 
        # which has a citytiles field that
        # contains the information of all citytiles in that city
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                dist = city_tile.pos.distance_to(pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_city_tile = city_tile
    return closest_city_tile


def find_city_tile_with_less_fuel(player):
    closest_city_tile = None
    if len(player.cities) > 0:
        less_fuel = math.inf
        # the cities are stored as a dictionary mapping city id to the city object, 
        # which has a citytiles field that
        # contains the information of all citytiles in that city
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                fuel = city.fuel
                if city.fuel < less_fuel:
                    less_fuel = city.fuel
                    less_fuel_city_tile = city_tile
    return less_fuel_city_tile


def find_nearest_position(unit, positions):
    nearest_position = None
    nearest_distance = None
    for position in positions:
        distance = abs(unit[0] - position[0]) + abs(unit[1] - position[1])
        if nearest_distance == None or distance < nearest_distance:
            nearest_distance = distance
            nearest_position = position
    nearest_point = Position(nearest_position[0],nearest_position[1])
    return nearest_point


def city_fuels(player):
    fuels = []
    if len(player.cities) > 0:
        for k, city in player.cities.items():
            #for city_tile in city.citytiles:
            fuels.append(city.fuel)
        return fuels
    return [0]


def direction_to_pos(x,y,direction):
    directions = {
        'e':np.array([+1,0]),
        'w':np.array([-1,0]),
        'n':np.array([0,-1]),
        's':np.array([0,+1]),
        'c':np.array([0,0]),
    }
    return tuple(np.array([x,y])+directions[direction])

def get_cooldown(player):
    cooldown = 0
    for k, city in player.cities.items():
        for city_tile in city.citytiles:
            if city_tile.cooldown>cooldown:
                cooldown = city_tile.cooldown
    return cooldown   


def get_window(unit):
    # agent window size is aways 12 and may be less than game field size
    window_size = 12
    fx = int(unit.pos.x-window_size/2 if unit.pos.x-window_size/2>=0 else 0)
    tx = int(fx+window_size)
    fy = int(unit.pos.y-window_size/2 if unit.pos.y-window_size/2>=0 else 0)
    ty = int(fy+window_size)
    return fx, tx, fy, ty



game_state = None
def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])    
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    width, height = game_state.map.width, game_state.map.height

    fields = {}
    fields['wood'] = np.zeros([width, height])
    fields['my_city_fuel'] = np.zeros([width, height])
    fields['my_city_cooldown'] = np.zeros([width, height])
    fields['my_city_light_upkeep'] = np.zeros([width, height])
    fields['op_city'] = np.zeros([width, height])
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource() and cell.resource.type == 'wood':
                fields[cell.resource.type][x][y] = cell.resource.amount
            if cell.citytile != None:
                if cell.citytile.team == observation.player:
                    fields['my_city_fuel'][x][y] = cell.citytile.fuel
                    fields['my_city_cooldown'][x][y] = cell.citytile.cooldown
                    fields['my_city_light_upkeep'][x][y] = cell.citytile.light_upkeep
                else:
                    fields['op_city'][x][y] = 1

    params = np.zeros(12*12)
    params[0] = game_state.map.width
    params[1] = observation["step"]

    for unit in player.units:
        # if the unit is a worker (can mine resources) and can perform an action this turn
        if unit.is_worker() and unit.can_act(): # and len(cities_x)>0:
            
            params[2] = 1 if unit.can_build(game_state.map) else 0
            params[3] = unit.get_cargo_space_left()

            fx,tx,fy,ty = get_window(unit)
            input_layer = np.concatenate([
                params,
                fields['wood'],
                fields['my_city_fuel'],
                fields['my_city_cooldown'],
                fields['my_city_light_upkeep'],
                fields['op_city'],
            ])            
            
            output_layer = np.array(6) # ToDo: apply RL there
            
            # build city
            if output_layer[5]>0.5:
                action = unit.build_city()
                actions.append(action)

            # move
            dir_max = np.max(output_layer[:5])
            if dir_max>0.5:
                direction_index = output_layer[:5].argmax()
                directions = [
                    DIRECTIONS.NORTH,
                    DIRECTIONS.EAST,
                    DIRECTIONS.SOUTH,
                    DIRECTIONS.WEST,
                    DIRECTIONS.CENTER
                    ]
                dir_to = directions[direction_index]
                action = unit.move(dir_to)
                actions.append(action)
    
    ### City tile: build new worker
    if len(player.cities.items()) > len(player.units):
        queued_units = 0        
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                if len(player.cities.items()) <= len(player.units) + queued_units:
                    break
                if city_tile.cooldown == 0:
                    actions.append(city_tile.build_worker())
                    queued_units += 1
    
    return actions
