# 🧠 Aprendizaje por Refuerzo con Flask – Proyecto Académico

Este proyecto implementa un entorno interactivo de **Aprendizaje por Refuerzo (Reinforcement Learning, RL)** utilizando **Flask**, **Q-Learning** y un entorno personalizado tipo **GridWorld 5x5**.

Incluye:

- Interfaz web moderna (Bootstrap + diseño minimalista)
- Entrenamiento configurable con Q-Learning
- Gráfica dinámica de recompensas por episodio
- Visualización de trayectoria del agente
- Síntesis teórica completa (conceptos, algoritmos, APA7)
- Estructura adecuada para repositorio académico

---

# 🎯 Objetivo del Proyecto

Comprender los fundamentos del Aprendizaje por Refuerzo y aplicarlos mediante la implementación de un agente capaz de aprender a tomar decisiones secuenciales. El proyecto permite:

- Definir un entorno RL simple.
- Configurear parámetros clave de aprendizaje.
- Entrenar un agente mediante Q-Learning.
- Observar las recompensas acumuladas.
- Visualizar la política aprendida.
- Integrar todo en una interfaz Flask.

---

# 📌 Contenido del Proyecto

## 1. Conceptos Básicos

Incluye teoría sobre:

- Qué es RL
- Comparación con supervisado y no supervisado
- Componentes: agente, entorno, estados, acciones, recompensas, política
- Explorar vs explotar (ε-greedy)
- Retorno acumulado y descuento temporal
- Algoritmos principales:
  - Q-Learning
  - SARSA
  - Deep Q-Network (DQN)
- Buenas prácticas:
  - Manejo de recompensas
  - Estabilidad del entrenamiento
  - Convergencia
  - Exploración adecuada

Se incluyen referencias APA 7.

---

## 2. Caso Práctico – GridWorld con Q-Learning

El entorno consta de:

- Grid 5x5
- Estado inicial: (0,0)
- Meta: (4,4)
- Obstáculos
- Recompensas:
  - -1 por movimiento
  - +10 al llegar a la meta
  - -10 por caer en obstáculo

### Parámetros ajustables en la interfaz:

| Parámetro | Descripción |
|----------|-------------|
| `episodes` | Número de episodios de entrenamiento |
| `max_steps` | Máx. pasos por episodio |
| `alpha` | Tasa de aprendizaje |
| `gamma` | Factor de descuento |
| `epsilon` | Exploración inicial |
| `epsilon_min` | Exploración mínima |
| `epsilon_decay` | Disminución progresiva de ε |

### Resultados generados:

- Archivo `q_table.pkl`
- Gráfica dinámica de recompensas
- Trayectoria del agente usando política greedy

---

# 🖥️ Interfaz Web Flask

La aplicación expone 2 secciones:

### ✔ Conceptos Básicos  
Explicación teórica completa (RL, algoritmos, APA).

### ✔ Caso Práctico  
Entrenamiento interactivo + visualizaciones:

- Entrenar agente
- Mostrar gráfica de recompensas
- Probar política aprendida
- Ver trayectorias en GridWorld

---

# 📂 Estructura del Proyecto
/static
/templates
base.html
index.html
rl_conceptos.html
rl_caso_practico.html
rl_gridworld.py
app.py
q_table.pkl (generado tras entrenamiento)
README.md
