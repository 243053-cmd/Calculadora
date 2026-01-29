from flask import Flask, render_template, request
from functions import solve_separable_raw, solve_exact_raw

app = Flask(__name__)

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/separable", methods=["GET"])
def separable_page():
    return render_template("separable.html")

@app.route("/exact", methods=["GET"])
def exact_page():
    return render_template("exact.html")

@app.route("/solve/separable", methods=["POST"])
def solve_separable():
    raw = request.form.get("equation", "")
    ok, method, solution, steps, error = solve_separable_raw(raw)
    return render_template("result.html", ok=ok, method=method, solution=solution, steps=steps, error=error)

@app.route("/solve/exact", methods=["POST"])
def solve_exact():
    raw = request.form.get("equation", "")
    ok, method, solution, steps, error = solve_exact_raw(raw)
    return render_template("result.html", ok=ok, method=method, solution=solution, steps=steps, error=error)

if __name__ == "__main__":
    app.run(debug=True)
