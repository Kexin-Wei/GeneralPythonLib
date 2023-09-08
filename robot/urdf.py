from dataclasses import dataclass
from typing import Union
from enum import Enum


class JointType(Enum):
    fixed = "fixed"
    revolute = "revolute"
    continuous = "continuous"
    prismatic = "prismatic"
    floating = "floating"
    planar = "planar"


@dataclass
class Origin:
    xyz: list[float]
    rpy: list[float]


@dataclass
class Inertia:
    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float


@dataclass
class Box:
    size: list[float]


@dataclass
class Cylinder:
    radius: float
    length: float


@dataclass
class Sphere:
    radius: float


@dataclass
class Mesh:
    filename: str
    scale: list[float]


@dataclass
class Inertial:
    origin: Origin
    mass: float
    inertia: Inertia


GEO_TYPE = Union[Box, Cylinder, Sphere, Mesh]


@dataclass
class Geometry:
    type: str
    geo_type: GEO_TYPE
    geo_properties: dict


@dataclass
class Material:
    name: str
    color: list[float]
    texture: str


@dataclass
class Visual:
    name: str
    origin: Origin
    geometry: Geometry
    material: Material


@dataclass
class Collision:
    name: str
    origin: Origin
    geometry: Geometry


@dataclass
class Link:
    name: str
    inertial: Inertial
    visual: Visual
    collision: Collision


@dataclass
class Calibration:
    rising: float
    falling: float


@dataclass
class Dynamics:
    damping: float
    friction: float


@dataclass
class Limit:
    lower: float
    upper: float
    effort: float
    velocity: float


@dataclass
class Mimic:
    joint: str
    multiplier: float
    offset: float


@dataclass
class SafetyController:
    k_velocity: float
    soft_lower_limit: float
    soft_upper_limit: float


@dataclass
class Joint:
    name: str
    type: str
    type_list = JointType
    origin: Origin
    parent: str
    child: str
    axis: list[float]
    calibration: Calibration
    dynamics: Dynamics
    limit: Limit
    mimic: Mimic
    safety_controller: SafetyController


@dataclass
class HardwareInterface:
    name: str
    interfaces: list[str]


@dataclass
class Actuator:
    name: str
    mechanical_reduction: float
    hardware_interface: HardwareInterface


@dataclass
class Transmission:
    name: str
    type: str
    joint: HardwareInterface
    actuator: Actuator


@dataclass
class Robot:
    name: str
    links: list[Link]
    joints: list[Joint]
    materials: list[Material]
    transmissions: list[Transmission]
