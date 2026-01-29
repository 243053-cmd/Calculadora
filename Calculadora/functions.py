import re
from sympy import symbols, Eq, diff, integrate, latex, simplify
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from sympy.core.sympify import SympifyError
from sympy.simplify.simplify import separatevars

# Variables
x, y = symbols("x y")
yp = symbols("yp")     # dy/dx
dx, dy = symbols("dx dy")

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# -------------------------
# Normalización / parseo
# -------------------------

def _normalize_common(s: str) -> str:
    """
    Normaliza:
      dy/dx -> yp
      y'    -> yp
      ^     -> **  (x^2 permitido)
    """
    s = (s or "").strip()
    s = s.replace("dy/dx", "yp")
    s = s.replace("y'", "yp")
    s = s.replace("^", "**")
    return s

def _insert_mul_before_differentials(s: str) -> str:
    """
    Inserta '*' antes de dx/dy cuando vienen pegados:
      (x+y)dx -> (x+y)*dx
      xdx -> x*dx
    """
    return re.sub(r'(?<=[0-9A-Za-z\)])\s*(d[xy])\b', r'*\1', s)

def safe_parse(expr_str: str, allow_differentials: bool = False):
    allowed = {"x": x, "y": y, "yp": yp}
    if allow_differentials:
        allowed.update({"dx": dx, "dy": dy})
    return parse_expr(expr_str, local_dict=allowed, transformations=TRANSFORMS, evaluate=True)

def parse_equation_input(raw: str) -> Eq:
    """
    Acepta:
      - dy/dx = ...
      - yp = ...
      - M*dx + N*dy = 0
      - M + N*yp = 0
      - expresión sin '=' (asume =0)
    """
    s = _normalize_common(raw)
    if not s:
        raise ValueError("Entrada vacía.")

    if "dx" in s or "dy" in s:
        s = _insert_mul_before_differentials(s)

    allow = ("dx" in s or "dy" in s)

    if "=" in s:
        left, right = s.split("=", 1)
        L = safe_parse(left, allow_differentials=allow)
        R = safe_parse(right, allow_differentials=allow)
        return Eq(L, R)

    expr = safe_parse(s, allow_differentials=allow)
    return Eq(expr, 0)

# -------------------------
# Exactas: extraer M y N
# -------------------------

def parse_exact_MN(raw: str):
    """
    Extrae (M, N) de:
      M*dx + N*dy = 0
    """
    s = _normalize_common(raw)
    if not s:
        raise ValueError("Entrada vacía.")

    s = _insert_mul_before_differentials(s)

    if "=" in s:
        left, right = s.split("=", 1)
        F = simplify(
            safe_parse(left, allow_differentials=True) -
            safe_parse(right, allow_differentials=True)
        )
    else:
        F = simplify(safe_parse(s, allow_differentials=True))

    M = simplify(F.coeff(dx))
    N = simplify(F.coeff(dy))
    resto = simplify(F - (M*dx + N*dy))

    if resto != 0:
        raise ValueError("Formato exacto inválido. Usa M*dx + N*dy = 0.")

    if M == 0 and N == 0:
        raise ValueError("No detecté términos con dx y dy.")

    return M, N

def to_slope_form(eq: Eq):
    """
    Convierte a yp = RHS(x,y) si se puede.
    Soporta:
      - yp = ...
      - ... = yp
      - M*dx + N*dy = 0 -> yp = -M/N
      - A*yp + B = 0
    """
    L = simplify(eq.lhs)
    R = simplify(eq.rhs)

    if L == yp:
        return simplify(R)
    if R == yp:
        return simplify(L)

    # Si aparecen dx/dy: yp = -M/N
    if (L.has(dx) or L.has(dy) or R.has(dx) or R.has(dy)):
        F = simplify(L - R)
        M = simplify(F.coeff(dx))
        N = simplify(F.coeff(dy))
        resto = simplify(F - (M*dx + N*dy))
        if resto == 0 and N != 0:
            return simplify(-M / N)

    # Lineal en yp: A*yp + B = 0
    F = simplify(L - R)
    A = simplify(diff(F, yp))
    if A != 0:
        B = simplify(F.subs(yp, 0))
        if simplify(F - (A*yp + B)) == 0:
            return simplify(-B / A)

    return None

def to_exact_from_equation(eq: Eq):
    """
    Obtiene M y N desde F = M + N*yp (lineal en yp).
    """
    F = simplify(eq.lhs - eq.rhs)
    N = simplify(diff(F, yp))
    if N == 0:
        return None, None

    M = simplify(F.subs(yp, 0))
    if simplify(F - (M + N*yp)) != 0:
        return None, None

    return M, N

# -------------------------
# Separables
# -------------------------

def _separate_rhs(rhs):
    """
    Detecta rhs = f(x)*g(y)
    """
    rhs_s = simplify(rhs)
    sep = separatevars(rhs_s, symbols=[x, y], dict=True, force=True)
    if not isinstance(sep, dict):
        return False, None, None

    fx = simplify(sep.get(x, 1))
    gy = simplify(sep.get(y, 1))

    if fx.has(y) or gy.has(x):
        return False, None, None

    return True, fx, gy

def solve_separable_from_rhs(rhs):
    steps = []
    ok, fx, gy = _separate_rhs(rhs)
    if not ok:
        return None, ["No se pudo separar en f(x)*g(y)."]

    rhs_l = latex(simplify(rhs))
    fx_l = latex(fx)
    gy_l = latex(gy)

    steps.append(rf"Partimos de la ecuación: \(\frac{{dy}}{{dx}} = {rhs_l}\)")
    steps.append(rf"Se identifica separación: \(f(x)={fx_l}\) y \(g(y)={gy_l}\)")
    steps.append(r"Separamos variables: \(\frac{1}{g(y)}\,dy = f(x)\,dx\)")

    left_int = integrate(1/gy, y)
    right_int = integrate(fx, x)

    left_l = latex(left_int)
    right_l = latex(right_int)

    steps.append(rf"Integramos ambos lados: \({left_l} = {right_l} + C\)")

    sol = rf"{left_l} = {right_l} + C"
    return sol, steps

def solve_separable_raw(raw: str):
    try:
        eq = parse_equation_input(raw)
        rhs = to_slope_form(eq)

        if rhs is None:
            return False, "Separable", None, [], "¡Ups! No es posible resolver esa ecuación con el método separable."

        sol, steps = solve_separable_from_rhs(rhs)
        if sol is None:
            return False, "Separable", None, [], "¡Ups! No es posible resolver esa ecuación con el método separable."

        return True, "Separable", sol, steps, None

    except (SympifyError, SyntaxError, TypeError) as e:
        return False, "Separable", None, [], f"Error al interpretar la ecuación: {e}"
    except ValueError as e:
        return False, "Separable", None, [], str(e)

# -------------------------
# Exactas
# -------------------------

def solve_exact_from_MN(M, N):
    steps = []
    M_s = simplify(M)
    N_s = simplify(N)

    steps.append(rf"Identificamos: \(M(x,y)={latex(M_s)}\) y \(N(x,y)={latex(N_s)}\)")

    My = simplify(diff(M_s, y))
    Nx = simplify(diff(N_s, x))

    steps.append(rf"Verificamos exactitud: \(\frac{{\partial M}}{{\partial y}}={latex(My)}\) y \(\frac{{\partial N}}{{\partial x}}={latex(Nx)}\)")

    if simplify(My - Nx) != 0:
        return None, ["La ecuación NO es exacta porque las derivadas parciales no coinciden."]

    F_partial = integrate(M_s, x)
    steps.append(rf"Integramos \(M\) respecto a \(x\): \(F(x,y)={latex(F_partial)}+h(y)\)")

    hprime = simplify(N_s - diff(F_partial, y))
    steps.append(rf"Calculamos \(h'(y)=N-\frac{{\partial}}{{\partial y}}(\int M\,dx)\): \(h'(y)={latex(hprime)}\)")

    h_of_y = integrate(hprime, y)
    steps.append(rf"Integramos \(h'(y)\): \(h(y)={latex(h_of_y)}\)")

    F = simplify(F_partial + h_of_y)
    steps.append(rf"Solución implícita: \({latex(F)}=C\)")

    sol = rf"{latex(F)} = C"
    return sol, steps

def solve_exact_raw(raw: str):
    try:
        # 1) Preferimos M*dx + N*dy = 0
        try:
            M, N = parse_exact_MN(raw)
        except ValueError:
            # 2) Intentamos M + N*yp = 0
            eq = parse_equation_input(raw)
            M, N = to_exact_from_equation(eq)

            if M is None or N is None:
                return False, "Exacta", None, [], "¡Ups! No es posible resolver esa ecuación con el método exacto."

        sol, steps = solve_exact_from_MN(M, N)
        if sol is None:
            return False, "Exacta", None, [], "¡Ups! No es posible resolver esa ecuación con el método exacto."

        return True, "Exacta", sol, steps, None

    except (SympifyError, SyntaxError, TypeError) as e:
        return False, "Exacta", None, [], f"Error al interpretar la ecuación: {e}"
    except ValueError as e:
        return False, "Exacta", None, [], str(e)
