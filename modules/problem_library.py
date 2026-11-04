"""A small starter library of solved problems, so the problem-set generator works with no history and no language model.
Each entry is (statement, model JSON in the same format the extraction step produces)."""


def _v(symbol: str, meaning: str, value, unit: str | None) -> dict:
    return {"symbol": symbol, "meaning": meaning, "known_value": None if value is None else str(value), "unit": unit}


def _eq(name: str, expression: str) -> dict:
    return {"name": name, "kind": "equation", "expression": expression, "derivation": "standard result"}


LIBRARY: dict[str, tuple[str, dict]] = {
    "Constant acceleration (kinematics)": (
        "A car accelerates uniformly from 8 m/s to 20 m/s in 6 seconds. Find its acceleration and the distance it travels.",
        {"problem_domain": "kinematics", "problem_type": "algebraic",
         "variables": [_v("u", "initial velocity", 8, "m/s"), _v("v", "final velocity", 20, "m/s"), _v("t", "time elapsed", 6, "s"),
                       _v("a", "acceleration", None, "m/s^2"), _v("d", "distance traveled", None, "m")],
         "equations": [_eq("final velocity equation", "Eq(v, a*t + u)"), _eq("displacement equation", "Eq(d, 0.5*a*t**2 + t*u)")],
         "solve_for": ["a", "d"], "assumptions": ["acceleration is uniform"]}),
    "Projectile range": (
        "A ball is launched at 20 m/s at an angle of 30 degrees above level ground. How far does it travel horizontally before landing?",
        {"problem_domain": "kinematics", "problem_type": "algebraic",
         "variables": [_v("v", "launch speed", 20, "m/s"), _v("theta", "launch angle", 30, "degrees"),
                       _v("g", "gravitational acceleration", 9.8, "m/s^2"), _v("R", "horizontal range", None, "m")],
         "equations": [_eq("range equation", "Eq(R, v**2*sin(2*theta*pi/180)/g)")],
         "solve_for": ["R"], "assumptions": ["flat ground", "no air resistance"]}),
    "Series circuit": (
        "A 12 V battery is connected across a 4 ohm resistor and a 6 ohm resistor in series. Find the current and the power the battery delivers.",
        {"problem_domain": "circuits", "problem_type": "algebraic",
         "variables": [_v("V", "battery voltage", 12, "V"), _v("R1", "first resistance", 4, "ohm"), _v("R2", "second resistance", 6, "ohm"),
                       _v("I", "current", None, "A"), _v("P", "power delivered", None, "W")],
         "equations": [_eq("Ohm's law for the series circuit", "Eq(V, I*(R1 + R2))"), _eq("electrical power", "Eq(P, V*I)")],
         "solve_for": ["I", "P"], "assumptions": ["ideal battery", "ideal wires"]}),
    "Pendulum period": (
        "A simple pendulum has a length of 1.5 m. What is its period for small oscillations?",
        {"problem_domain": "oscillations", "problem_type": "algebraic",
         "variables": [_v("L", "pendulum length", 1.5, "m"), _v("g", "gravitational acceleration", 9.8, "m/s^2"), _v("T", "period", None, "s")],
         "equations": [_eq("pendulum period", "Eq(T, 2*pi*sqrt(L/g))")],
         "solve_for": ["T"], "assumptions": ["small angle", "massless string"]}),
    "Kinetic energy and momentum": (
        "A 1200 kg car travels at 25 m/s. Find its kinetic energy and its momentum.",
        {"problem_domain": "mechanics", "problem_type": "algebraic",
         "variables": [_v("m", "mass", 1200, "kg"), _v("v", "speed", 25, "m/s"), _v("KE", "kinetic energy", None, "J"),
                       _v("p", "momentum", None, "kg*m/s")],
         "equations": [_eq("kinetic energy", "Eq(KE, 0.5*m*v**2)"), _eq("momentum", "Eq(p, m*v)")],
         "solve_for": ["KE", "p"], "assumptions": []}),
    "Ideal gas": (
        "2.0 mol of an ideal gas at 300 K occupies a volume of 0.050 m^3. What is its pressure? Use R = 8.314 J/(mol K).",
        {"problem_domain": "thermodynamics", "problem_type": "algebraic",
         "variables": [_v("n", "amount of gas", 2.0, "mol"), _v("T", "temperature", 300, "K"), _v("V", "volume", 0.05, "m^3"),
                       _v("R", "gas constant", 8.314, "J/(mol*K)"), _v("P", "pressure", None, "Pa")],
         "equations": [_eq("ideal gas law", "Eq(P*V, n*R*T)")],
         "solve_for": ["P"], "assumptions": ["ideal gas"]}),
    "Compound interest": (
        "$1000 is invested at 5% annual interest, compounded 12 times per year. How much is it worth after 10 years?",
        {"problem_domain": "finance", "problem_type": "algebraic",
         "variables": [_v("P", "principal", 1000, "dollars"), _v("r", "annual interest rate", 0.05, None),
                       _v("n", "compounding periods per year", 12, None), _v("t", "years", 10, "years"), _v("A", "final amount", None, "dollars")],
         "equations": [_eq("compound interest", "Eq(A, P*(1 + r/n)**(n*t))")],
         "solve_for": ["A"], "assumptions": ["no deposits or withdrawals"]}),
}
