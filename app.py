from flask import Flask, render_template, request, jsonify
from rl_gridworld import train_agent, run_greedy_episode

app = Flask(__name__)

# ============================
# RUTAS PRINCIPALES
# ============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/rl/conceptos")
def rl_conceptos():
    return render_template("rl_conceptos.html")

@app.route("/rl/caso-practico")
def rl_caso_practico():
    return render_template("rl_caso_practico.html")


# ============================
# API PARA ENTRENAR RL
# ============================

@app.route("/api/rl/train", methods=["POST"])
def api_train():
    data = request.get_json() or {}

    episodes = int(data.get("episodes", 500))
    max_steps = int(data.get("max_steps", 100))
    alpha = float(data.get("alpha", 0.1))
    gamma = float(data.get("gamma", 0.99))
    epsilon = float(data.get("epsilon", 1.0))
    epsilon_min = float(data.get("epsilon_min", 0.01))
    epsilon_decay = float(data.get("epsilon_decay", 0.995))

    rewards, epsilons, _ = train_agent(
        episodes=episodes,
        max_steps=max_steps,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        save_path="q_table.pkl"
    )

    return jsonify({
        "episodes": episodes,
        "rewards": rewards,
        "epsilons": epsilons,
        "avg_reward_last_50": float(sum(rewards[-50:]) / min(50, len(rewards)))
    })


# ============================
# API PARA PROBAR POLÍTICA
# ============================

@app.route("/api/rl/run_episode", methods=["GET"])
def api_run_episode():
    sim = run_greedy_episode(max_steps=50, q_path="q_table.pkl")
    if sim is None:
        return jsonify({"error": "No hay modelo entrenado"}), 400
    return jsonify(sim)


if __name__ == "__main__":
    app.run(debug=True)
