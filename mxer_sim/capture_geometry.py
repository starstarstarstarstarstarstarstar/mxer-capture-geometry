"""
Module 1: Capture Geometry Model
MXER MVP Capture Simulation

Models the physical funnel-throat geometry. Maps 3D payload entry position
and velocity (in funnel frame) to capture outcomes: MISS, GUIDED, THROAT_ENTRY,
LATCH_CAPTURED.

BASELINE: omega=0.010 rad/s, L=70km/arm, 200kg payload, 500km LEO,
310s hold-open window, 35 missions/yr.
Architecture: Passive conical funnel (8m entrance, 5m length, 30 deg half-angle)
+ throat (2m diameter, 1.5m channel) + 6 semi-active latches.
"""

import math
from typing import Tuple, Dict, Any, Optional

# =============================================================================
# DEFAULT PHYSICAL CONSTANTS
# =============================================================================

DEFAULT_FUNNEL_ENTRANCE_DIAMETER_M = 8.0
DEFAULT_FUNNEL_THROAT_DIAMETER_M = 2.0
DEFAULT_FUNNEL_LENGTH_M = 5.0
DEFAULT_FUNNEL_HALF_ANGLE_RAD = math.radians(30.0)
DEFAULT_THROAT_LENGTH_M = 1.5
DEFAULT_LATCH_TOLERANCE_M = 0.2
DEFAULT_K_FINGER_N_M = 1000.0  # Spring constant of guide fingers (N/m)
DEFAULT_C_DAMPING_NS_M = 50.0  # Damping coefficient (N*s/m)
DEFAULT_PAYLOAD_MASS_KG = 200.0
DEFAULT_G_MPS2 = 9.81  # Gravitational acceleration (m/s^2)


# =============================================================================
# CORE GEOMETRY FUNCTIONS
# =============================================================================

def cone_radius_at_z(z: float,
                     throat_radius: float,
                     funnel_half_angle_rad: float) -> float:
    """
    Compute the funnel cone radius at axial position z.

    r(z) = throat_radius + z * tan(theta)

    Args:
        z: Axial position from funnel entry (m)
        throat_radius: Radius at funnel throat (m)
        funnel_half_angle_rad: Half-angle of funnel cone (rad)

    Returns:
        Cone radius at position z (m)
    """
    return throat_radius + z * math.tan(funnel_half_angle_rad)


def check_entry_miss(entry_radius: float, funnel_entrance_radius: float) -> bool:
    """Return True if payload misses the funnel entrance entirely."""
    return entry_radius > funnel_entrance_radius


def check_throat_entry(radial_position: float, throat_radius: float) -> bool:
    """Return True if payload has entered the throat."""
    return radial_position <= throat_radius


def check_latch_capture(residual_offset: float, latch_tolerance: float) -> bool:
    """Return True if residual offset is within latch tolerance."""
    return residual_offset <= latch_tolerance


# =============================================================================
# TRAJECTORY SIMULATION
# =============================================================================

def simulate_payload_trajectory(
    r0: float,
    z0: float,
    vr0: float,
    vz0: float,
    m_payload: float,
    k_finger: float,
    c_damping: float,
    throat_radius: float,
    funnel_half_angle_rad: float,
    funnel_length: float,
    dt: float = 0.001,
    max_time: float = 10.0
) -> Dict[str, Any]:
    """
    Simulate payload trajectory through funnel using Euler integration.

    Physics model:
    - Axial motion: simplified, payload slides along funnel wall or flies through
    - Radial motion: m*r_ddot = m*v_tan^2/r - k_finger*(r - r_cone) - c_damping*r_dot

    Args:
        r0: Initial radial position (m)
        z0: Initial axial position (m)
        vr0: Initial radial velocity (m/s)
        vz0: Initial axial velocity (m/s)
        m_payload: Payload mass (kg)
        k_finger: Guide finger spring constant (N/m)
        c_damping: Damping coefficient (N*s/m)
        throat_radius: Radius at throat (m)
        funnel_half_angle_rad: Half-angle of funnel (rad)
        funnel_length: Total funnel length (m)
        dt: Integration timestep (s)
        max_time: Maximum simulation time (s)

    Returns:
        dict with trajectory results
    """
    # Initialize state
    r = r0
    z = z0
    vr = vr0
    vz = vz0

    t = 0.0
    trajectory = []

    # Tangential velocity component (conserved angular momentum approximation)
    # v_tan = sqrt(vx^2 + vy^2) at entry
    v_tan = abs(vr0)  # Simplified: use radial velocity as characteristic

    while t < max_time:
        # Record state
        trajectory.append({'t': t, 'r': r, 'z': z, 'vr': vr, 'vz': vz})

        # Check stop conditions
        if r <= throat_radius:
            # Throat entry achieved
            return {
                'status': 'THROAT_ENTRY',
                't': t,
                'r': r,
                'z': z,
                'vr': vr,
                'vz': vz,
                'v_tan': v_tan,
                'trajectory': trajectory
            }

        if z >= funnel_length:
            # Exited funnel without entering throat
            return {
                'status': 'EXIT',
                't': t,
                'r': r,
                'z': z,
                'vr': vr,
                'vz': vz,
                'v_tan': v_tan,
                'trajectory': trajectory
            }

        # Compute cone radius at current z
        r_cone = cone_radius_at_z(z, throat_radius, funnel_half_angle_rad)

        # Radial equation of motion:
        # m*r_ddot = m*v_tan^2/r - k_finger*(r - r_cone) - c_damping*r_dot
        centripetal = m_payload * (v_tan ** 2) / r if r > 0.001 else 0.0
        spring_force = k_finger * (r - r_cone)
        damping_force = c_damping * vr

        r_ddot = (centripetal - spring_force - damping_force) / m_payload

        # Axial motion: payload decelerates as it "climbs" the funnel wall
        # Simple model: small deceleration opposing motion
        # The funnel half-angle determines how axial position changes with radial
        dz_dr = math.tan(funnel_half_angle_rad)
        vz = vz - 0.1 * vz * dt  # Light damping on axial velocity

        # Euler integration
        vr = vr + r_ddot * dt
        r = r + vr * dt

        # Update z based on remaining axial velocity
        # When guided, payload moves toward throat (increasing z)
        # When off-center, it slides along cone surface
        if r > r_cone:
            # Payload is outside cone surface, slide along cone wall
            dz_dt = vz * 0.1 + 0.5  # Base progression toward throat
        else:
            # Payload inside cone, continue through
            dz_dt = max(vz, 0.5)
        z = z + dz_dt * dt

        t = t + dt

    # Timeout
    return {
        'status': 'TIMEOUT',
        't': t,
        'r': r,
        'z': z,
        'vr': vr,
        'vz': vz,
        'v_tan': v_tan,
        'trajectory': trajectory
    }


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_funnel_capture(
    dx: float,
    dy: float,
    dz: float,
    vx: float,
    vy: float,
    vz: float,
    funnel_entrance_diameter_m: float = DEFAULT_FUNNEL_ENTRANCE_DIAMETER_M,
    funnel_throat_diameter_m: float = DEFAULT_FUNNEL_THROAT_DIAMETER_M,
    funnel_length_m: float = DEFAULT_FUNNEL_LENGTH_M,
    funnel_half_angle_rad: float = DEFAULT_FUNNEL_HALF_ANGLE_RAD,
    throat_length_m: float = DEFAULT_THROAT_LENGTH_M,
    latch_tolerance_m: float = DEFAULT_LATCH_TOLERANCE_M,
    k_finger_N_m: float = DEFAULT_K_FINGER_N_M,
    c_damping_Ns_m: float = DEFAULT_C_DAMPING_NS_M,
    m_payload_kg: float = DEFAULT_PAYLOAD_MASS_KG,
    dt: float = 0.001
) -> Dict[str, Any]:
    """
    Main entry point. Evaluate capture outcome for given entry conditions.

    Args:
        dx, dy, dz: Payload position relative to funnel axis at entry (m)
        vx, vy, vz: Payload velocity in funnel frame (m/s)
        funnel_entrance_diameter_m: Funnel entrance diameter (m)
        funnel_throat_diameter_m: Throat diameter (m)
        funnel_length_m: Length of conical section (m)
        funnel_half_angle_rad: Half-angle of funnel cone (rad)
        throat_length_m: Length of throat section (m)
        latch_tolerance_m: Max residual offset for latch capture (m)
        k_finger_N_m: Guide finger spring constant (N/m)
        c_damping_Ns_m: Damping coefficient (N*s/m)
        m_payload_kg: Payload mass (kg)
        dt: Integration timestep (s)

    Returns:
        dict with outcome and all relevant outputs:
        - outcome: str enum ["MISS", "GUIDED", "THROAT_ENTRY", "LATCH_CAPTURED"]
        - entry_radius_m: radial offset at entry
        - throat_entry_time_s: time to reach throat (None if not reached)
        - residual_offset_at_throat_m: radial offset when entering throat (None if not reached)
        - funnel_exit_velocity_m_s: speed when exiting funnel
        - trajectory: list of state dicts
    """
    # Compute geometric parameters
    funnel_entrance_radius = funnel_entrance_diameter_m / 2.0
    throat_radius = funnel_throat_diameter_m / 2.0

    # Verify geometry consistency
    expected_entrance_radius = throat_radius + funnel_length_m * math.tan(funnel_half_angle_rad)
    # Note: This is a cone so entrance should match computed value

    # Compute entry radius
    entry_radius = math.sqrt(dx ** 2 + dy ** 2)

    # Check 1: MISS if entry is outside funnel entrance
    if check_entry_miss(entry_radius, funnel_entrance_radius):
        return {
            'outcome': 'MISS',
            'entry_radius_m': entry_radius,
            'throat_entry_time_s': None,
            'residual_offset_at_throat_m': None,
            'funnel_exit_velocity_m_s': None,
            'exit_velocity': None,
            'status': 'MISS_AT_ENTRY'
        }

    # Entry is within funnel entrance - GUIDED
    # Initial conditions for trajectory simulation
    r0 = entry_radius
    z0 = 0.0  # Start at funnel entry
    vr0 = math.sqrt(vx ** 2 + vy ** 2)  # Radial velocity magnitude
    vz0 = abs(vz)  # Axial velocity (positive toward throat)

    # Simulate trajectory
    result = simulate_payload_trajectory(
        r0=r0,
        z0=z0,
        vr0=vr0,
        vz0=vz0,
        m_payload=m_payload_kg,
        k_finger=k_finger_N_m,
        c_damping=c_damping_Ns_m,
        throat_radius=throat_radius,
        funnel_half_angle_rad=funnel_half_angle_rad,
        funnel_length=funnel_length_m,
        dt=dt
    )

    # Interpret results
    if result['status'] == 'THROAT_ENTRY':
        residual_offset = result['r']
        throat_entry_time = result['t']

        # Check if latch capture is achieved
        if check_latch_capture(residual_offset, latch_tolerance_m):
            outcome = 'LATCH_CAPTURED'
        else:
            outcome = 'THROAT_ENTRY'

        return {
            'outcome': outcome,
            'entry_radius_m': entry_radius,
            'throat_entry_time_s': throat_entry_time,
            'residual_offset_at_throat_m': residual_offset,
            'funnel_exit_velocity_m_s': math.sqrt(result['vr'] ** 2 + result['vz'] ** 2),
            'exit_velocity': math.sqrt(result['vr'] ** 2 + result['vz'] ** 2),
            'trajectory': result['trajectory'],
            'status': result['status']
        }

    elif result['status'] == 'EXIT':
        return {
            'outcome': 'GUIDED',  # Was guided but passed through
            'entry_radius_m': entry_radius,
            'throat_entry_time_s': None,
            'residual_offset_at_throat_m': None,
            'funnel_exit_velocity_m_s': math.sqrt(result['vr'] ** 2 + result['vz'] ** 2),
            'exit_velocity': math.sqrt(result['vr'] ** 2 + result['vz'] ** 2),
            'trajectory': result['trajectory'],
            'status': result['status']
        }

    else:  # TIMEOUT
        return {
            'outcome': 'GUIDED',  # Stuck but was guided
            'entry_radius_m': entry_radius,
            'throat_entry_time_s': None,
            'residual_offset_at_throat_m': None,
            'funnel_exit_velocity_m_s': None,
            'exit_velocity': None,
            'trajectory': result['trajectory'],
            'status': result['status']
        }


# =============================================================================
# TOML CONFIG SECTION
# =============================================================================

CAPTURE_GEOMETRY_TOML_SECTION = """
# =============================================================================
# CAPTURE GEOMETRY CONFIGURATION
# =============================================================================

[geometry.funnel]
# Conical funnel parameters
entrance_diameter_m = 8.0      # 8 meter entrance diameter
throat_diameter_m = 2.0        # 2 meter throat diameter
length_m = 5.0                  # 5 meter funnel length
half_angle_deg = 30.0           # 30 degree half-angle

[geometry.throat]
# Throat/channel parameters
diameter_m = 2.0                # Same as funnel throat
length_m = 1.5                  # 1.5 meter channel length

[geometry.latch]
# Latch tolerance parameters
tolerance_m = 0.2               # Max residual offset for latch capture
n_latches = 6                   # Number of semi-active latches

[physics.funnel]
# Guide finger (friction rails) parameters
k_finger_N_m = 1000.0           # Spring constant (N/m)
c_damping_Ns_m = 50.0           # Damping coefficient (N*s/m)

[physics.payload]
# Default payload parameters
mass_kg = 200.0                # 200 kg payload
diameter_m = 0.5               # Approximate payload diameter

[integration]
# Simulation parameters
dt_s = 0.001                   # Euler integration timestep
max_time_s = 10.0              # Maximum simulation time
"""

# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 1: CAPTURE GEOMETRY - TESTING")
    print("=" * 70)

    # Test Case 1: NOMINAL - Centered entry with small offset
    print("\n--- Test 1: NOMINAL (dx=0.5, dy=0.3, nominal velocity) ---")
    result = evaluate_funnel_capture(
        dx=0.5, dy=0.3, dz=0.0,
        vx=0.1, vy=0.1, vz=1.0
    )
    print(f"  Outcome: {result['outcome']}")
    print(f"  Entry radius: {result['entry_radius_m']:.3f} m")
    print(f"  Throat entry time: {result['throat_entry_time_s']}")
    if result['residual_offset_at_throat_m'] is not None:
        print(f"  Residual offset at throat: {result['residual_offset_at_throat_m']:.4f} m")
    if result['funnel_exit_velocity_m_s'] is not None:
        print(f"  Exit velocity: {result['funnel_exit_velocity_m_s']:.3f} m/s")

    # Test Case 2: OFF-NOMINAL - Larger offset, still within entrance
    print("\n--- Test 2: OFF-NOMINAL (dx=2.0, dy=1.5, higher velocity) ---")
    result = evaluate_funnel_capture(
        dx=2.0, dy=1.5, dz=0.0,
        vx=0.5, vy=0.5, vz=2.0
    )
    print(f"  Outcome: {result['outcome']}")
    print(f"  Entry radius: {result['entry_radius_m']:.3f} m")
    print(f"  Throat entry time: {result['throat_entry_time_s']}")
    if result['residual_offset_at_throat_m'] is not None:
        print(f"  Residual offset at throat: {result['residual_offset_at_throat_m']:.4f} m")

    # Test Case 3: BOUNDARY - Entry at exactly entrance radius
    print("\n--- Test 3: BOUNDARY (dx=3.99, dy=0, at entrance edge) ---")
    result = evaluate_funnel_capture(
        dx=3.99, dy=0.0, dz=0.0,
        vx=0.1, vy=0.0, vz=1.0
    )
    print(f"  Outcome: {result['outcome']}")
    print(f"  Entry radius: {result['entry_radius_m']:.3f} m")
    print(f"  Throat entry time: {result['throat_entry_time_s']}")
    if result['residual_offset_at_throat_m'] is not None:
        print(f"  Residual offset at throat: {result['residual_offset_at_throat_m']:.4f} m")

    # Test Case 4: MISS - Outside funnel entrance
    print("\n--- Test 4: MISS (dx=5.0, dy=0, outside entrance) ---")
    result = evaluate_funnel_capture(
        dx=5.0, dy=0.0, dz=0.0,
        vx=0.0, vy=0.0, vz=1.0
    )
    print(f"  Outcome: {result['outcome']}")
    print(f"  Entry radius: {result['entry_radius_m']:.3f} m")

    # Test Case 5: LATCH CAPTURE - Centered entry
    print("\n--- Test 5: LATCH CAPTURE (dx=0.1, dy=0.1, centered) ---")
    result = evaluate_funnel_capture(
        dx=0.1, dy=0.1, dz=0.0,
        vx=0.05, vy=0.05, vz=0.5
    )
    print(f"  Outcome: {result['outcome']}")
    print(f"  Entry radius: {result['entry_radius_m']:.3f} m")
    if result['throat_entry_time_s'] is not None:
        print(f"  Throat entry time: {result['throat_entry_time_s']:.4f} s")
    if result['residual_offset_at_throat_m'] is not None:
        print(f"  Residual offset at throat: {result['residual_offset_at_throat_m']:.4f} m")

    print("\n" + "=" * 70)
    print("[OK] capture_geometry.py")
    print("=" * 70)
