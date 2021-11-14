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


game_state = None
def agent(observation, configuration):
    global game_state
    city_fuel_limit = 300
    city_count_max_limit = 3

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
    #opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height
       
    fields = {}
    ### city
    # define city field
    fields['my_city'] = np.zeros([width, height])
    if len(player.cities) > 0:
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                fields['my_city'][city_tile.pos.x][city_tile.pos.y] = 1

    # all units
    #for any_player in game_state.players:
                
    # resources
    fields['wood'] = np.zeros([width, height])
    fields['coal'] = np.zeros([width, height])
    fields['uranium'] = np.zeros([width, height])
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():            
                fields[cell.resource.type][x][y] = 1
                
    free_tiles = fields['my_city']+fields['wood']+fields['coal']+fields['uranium']    
    # find new city position
    cities_x = np.where(fields['my_city']==1)[0]
    cities_y = np.where(fields['my_city']==1)[1]
    new_city_pos_list = []
    for i in range(len(cities_x)):
        x = cities_x[i]
        y = cities_y[i]
        if x > 0 and free_tiles[x-1][y] == 0:
            new_city_pos_list.append((x-1,y))
        if y > 0 and free_tiles[x][y-1] == 0:
            new_city_pos_list.append((x,y-1))
        if x < len(free_tiles)-1 and free_tiles[x+1][y] == 0:
            new_city_pos_list.append((x+1,y))
        if y < len(free_tiles[0])-1 and free_tiles[x][y+1] == 0:
            new_city_pos_list.append((x,y+1))
    
    city_cooldown = get_cooldown(player)
    
    ### unit
    resource_tiles = find_resources(game_state)
    queued_unit_tiles = []
    min_city_fuel = min(city_fuels(player))
    
    """for unit in player.units:                
        if unit.is_worker():
            queued_unit_tiles.append((unit.pos.x,unit.pos.y))"""
    
    for unit in player.units:        
        # if the unit is a worker (can mine resources) and can perform an action this turn
        if unit.is_worker() and unit.can_act() and len(cities_x)>0:
            
            # build new city
            new_city_pos = find_nearest_position([unit.pos.x, unit.pos.y], new_city_pos_list)
            if len(cities_x)<city_count_max_limit \
            and min_city_fuel>city_fuel_limit*len(cities_x) \
            and unit.can_build(game_state.map) \
            and (unit.pos.x, unit.pos.y) == (new_city_pos.x, new_city_pos.y):
                action = unit.build_city()
                actions.append(action)
                queued_unit_tiles.append((unit.pos.x,unit.pos.y))
                continue
                
            # we want to mine only if there is space left in the worker's cargo
            if unit.get_cargo_space_left() > 0:
                # find the closest resource if it exists to this unit
                closest_resource_tile = find_closest_resources(unit.pos, player, resource_tiles)
                if closest_resource_tile is not None:
                    # create a move action to move this unit in the direction of the closest resource tile and add to our actions list
                    dir_to = unit.pos.direction_to(closest_resource_tile.pos)
                    tile_to_queue = direction_to_pos(unit.pos.x,unit.pos.y,dir_to)
                    if not tile_to_queue in queued_unit_tiles:
                        action = unit.move(dir_to)
                        actions.append(action)
                        queued_unit_tiles.append(tile_to_queue)
                    else:
                        queued_unit_tiles.append((unit.pos.x,unit.pos.y))
            else:                
                if min_city_fuel>city_fuel_limit*len(cities_x) and len(new_city_pos_list):                    
                    # go to build a new city
                    
                    dir_to = unit.pos.direction_to(new_city_pos)
                    tile_to_queue = direction_to_pos(unit.pos.x,unit.pos.y,dir_to)
                    if not tile_to_queue in queued_unit_tiles:
                        action = unit.move(dir_to)
                        actions.append(action)
                        queued_unit_tiles.append(tile_to_queue)
                    else:
                        queued_unit_tiles.append((unit.pos.x,unit.pos.y))
                else:                
                    # find the closest citytile and move the unit towards it to drop resources to a citytile to fuel the city
                    closest_city_tile = find_closest_city_tile(unit.pos, player)
                    if closest_city_tile is not None:
                        # create a move action to move this unit in the direction of the closest resource tile and add to our actions list
                        dir_to = unit.pos.direction_to(closest_city_tile.pos)
                        tile_to_queue = direction_to_pos(unit.pos.x,unit.pos.y,dir_to)
                        if not tile_to_queue in queued_unit_tiles:
                            action = unit.move(dir_to)
                            actions.append(action)
                            queued_unit_tiles.append(tile_to_queue)
                        else:
                            queued_unit_tiles.append((unit.pos.x,unit.pos.y))
    
    ### City tile: build new worker                            
    if len(cities_x) > len(player.units) and city_cooldown==0:
        queued_units = 0        
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                if len(cities_x) <= len(player.units) + queued_units:
                    break
                actions.append(city_tile.build_worker())
                queued_units += 1
    
    return actions
