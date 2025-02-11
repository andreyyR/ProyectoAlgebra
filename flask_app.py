#!/usr/bin/env python3
from flask import Flask, render_template, request, url_for, redirect
import copy
import sympy
from sympy import sympify, simplify, together, sstr

app = Flask(__name__)

# =============================================================================
# Funciones lógicas (idénticas a las de tu código original, con algunos ajustes)
# =============================================================================

# Lista de nombres de variables para la notación de las incógnitas
variables = ['x', 'y', 'z', 'w', 'v', 'u', 'p', 'q']

def parse_expr_input(s):
    """
    Convierte la cadena s a una expresión simbólica usando sympy.
    Se reemplaza el operador '^' por '**'. Si la cadena está vacía se devuelve 0.
    Si la entrada es inválida se lanza excepción.
    """
    s = s.strip()
    if not s:
        return sympify("0")
    s = s.replace("^", "**")
    try:
        return sympify(s)
    except Exception as e:
        raise ValueError(f"Expresión inválida: '{s}'. Error: {e}")

def formatear_expr(expr):
    """
    Simplifica y agrupa la expresión para luego devolver una representación
    en una sola línea usando sstr.
    """
    expr = simplify(expr)
    expr = together(expr)
    return sstr(expr)

def formatear_matriz(mat):
    """
    Devuelve un string con la matriz aumentada formateada en una sola línea
    para cada fila (por ejemplo: [ a   1   1   1 ]).
    """
    lines = []
    for fila in mat:
        elems = [formatear_expr(x) for x in fila]
        line = "[ " + "   ".join(elems) + " ]"
        lines.append(line)
    return "\n".join(lines)

def formatear_sistema_equaciones(mat):
    """
    Dada la matriz aumentada, devuelve un string que muestra el sistema
    de ecuaciones en formato legible.
    """
    system_str = ""
    num_eq = len(mat)
    num_vars = len(mat[0]) - 1
    for i in range(num_eq):
        terms = []
        for j in range(num_vars):
            coef = simplify(mat[i][j])
            var = variables[j]
            term = f"{formatear_expr(coef)}*{var}"
            terms.append(term)
        eq = " + ".join(terms) + " = " + formatear_expr(mat[i][-1])
        system_str += eq + "\n"
    return system_str.strip()

def eliminacion_sin_pivote_simbolica(matriz_original):
    """
    Realiza la eliminación gaussiana sobre la matriz aumentada y genera un reporte
    detallado de cada paso.
    """
    matriz = copy.deepcopy(matriz_original)
    steps = []
    def add(line=""):
        steps.append(line)

    # Encabezado y sistema inicial
    add("Sistema de ecuaciones:")
    add(formatear_sistema_equaciones(matriz))
    add("")
    add("Solución por Eliminación Gaussiana:")
    add("")
    add("Convertimos la matriz aumentada a la forma escalonada por filas:")
    add(formatear_matriz(matriz))
    add("")

    n = len(matriz)
    condiciones = []  # Acumula condiciones necesarias (pivote ≠ 0, etc.)

    # Paso 1: Selección del pivote y eliminación hacia adelante
    for i in range(n):
        # Se busca en la columna i, desde la fila i en adelante, si existe un 1
        pivot_val = simplify(matriz[i][i])
        best_row = i
        for k in range(i, n):
            if simplify(matriz[k][i]) == 1:
                best_row = k
                break
        if best_row != i:
            matriz[i], matriz[best_row] = matriz[best_row], matriz[i]
            add(f"Intercambiamos R{i+1} y R{best_row+1}:")
            add(formatear_matriz(matriz))
            add("")
            pivot_val = simplify(matriz[i][i])

        # Si el pivote es 0, se intenta intercambiar con otra fila que tenga un valor distinto de 0
        if pivot_val == 0:
            for k in range(i+1, n):
                if simplify(matriz[k][i]) != 0:
                    best_row = k
                    break
            if best_row != i:
                matriz[i], matriz[best_row] = matriz[best_row], matriz[i]
                add(f"Intercambiamos R{i+1} y R{best_row+1} (pivote cero):")
                add(formatear_matriz(matriz))
                add("")
                pivot_val = simplify(matriz[i][i])
            else:
                add(f"Advertencia: No se encontró pivote distinto de cero en la columna {i+1}.")
                continue
        condiciones.append(f"{formatear_expr(pivot_val)}≠0")

        # Eliminación hacia adelante: anular los coeficientes de las filas inferiores
        for k in range(i+1, n):
            current = simplify(matriz[k][i])
            if pivot_val == 0:
                continue
            factor = simplify(current / pivot_val)
            add(f"Restamos ({formatear_expr(factor)})*R{i+1} de R{k+1}:")
            for c in range(i, n+1):
                matriz[k][c] = simplify(matriz[k][c] - factor * matriz[i][c])
            add(formatear_matriz(matriz))
            add("Condiciones acumuladas: " + ", ".join(condiciones))
            add("")

    add("Forma escalonada (row echelon form):")
    add(formatear_matriz(matriz))
    add("")

    # Paso 2: Sustitución hacia atrás para obtener la solución
    add("Resolviendo el sistema (sustitución hacia atrás):")
    solucion = [None] * n
    for i in range(n-1, -1, -1):
        suma = matriz[i][n]
        for j in range(i+1, n):
            suma = simplify(suma - matriz[i][j] * solucion[j])
        add(f"De la ecuación {i+1}:")
        add(f"{formatear_expr(matriz[i][i])}*{variables[i]} = {formatear_expr(suma)}")
        if simplify(matriz[i][i]) == 0:
            add("  (División por cero → solución indeterminada o infinita)")
            solucion[i] = sympy.symbols(f"{variables[i]}")
        else:
            solucion[i] = simplify(suma / matriz[i][i])
        add(f"Solución: {variables[i]} = {formatear_expr(solucion[i])}")
        add("")

    add("Respuesta final:")
    for i in range(n):
        add(f"  {variables[i]} = {formatear_expr(solucion[i])}")
    add("Solución general: X = (" + ", ".join(formatear_expr(sol) for sol in solucion) + ")")

    return "\n".join(steps)

# =============================================================================
# Rutas de la aplicación Flask (Interfaz Web)
# =============================================================================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Se asume un sistema de 3 ecuaciones con 3 incógnitas (3x3)
        try:
            rows = int(request.form.get("rows", 3))
            cols = int(request.form.get("cols", 3))  # número de incógnitas
        except:
            rows = 3
            cols = 3

        # Se construye la matriz aumentada a partir de los campos del formulario.
        matriz = []
        for i in range(rows):
            fila = []
            # Cada fila tiene (cols + 1) celdas: coeficientes y término independiente
            for j in range(cols + 1):
                cell_name = f"cell_{i}_{j}"
                cell_value = request.form.get(cell_name, "").strip()
                try:
                    expr = parse_expr_input(cell_value)
                except Exception as e:
                    return f"Error en la celda ({i+1}, {j+1}): {str(e)}"
                fila.append(expr)
            matriz.append(fila)

        try:
            reporte = eliminacion_sin_pivote_simbolica(matriz)
        except Exception as e:
            return f"Error al procesar el sistema: {str(e)}"

        return render_template("result.html", steps=reporte)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
