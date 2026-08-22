from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.config["SECRET_KEY"] = "arogyaflow-demo-key"


def phc_name_from_login(value):
    """Create a readable PHC label for the demo login identifier."""
    identifier = value.split("@", 1)[0].replace("_", " ").replace("-", " ").strip()
    name = " ".join(part.capitalize() for part in identifier.split()) or "My PHC"
    return name if "phc" in name.lower() else f"{name} PHC"


@app.context_processor
def logged_in_phc():
    return {
        "current_phc": session.get("phc_name", ""),
        "current_manager": session.get("manager_name", "PHC Manager"),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["phc_name"] = phc_name_from_login(request.form.get("email", ""))
        session["manager_name"] = "PHC Manager"
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        return redirect(url_for("dashboard"))
    return render_template("register.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/phc-network")
def phc_network():
    return render_template("phc_network.html")


@app.route("/medicine-inventory")
def medicine_inventory():
    return render_template("medicine_inventory.html")


@app.route("/beds-overview")
def beds_overview():
    return render_template("beds_overview.html")


@app.route("/medical-staff")
def medical_staff():
    return render_template("medical_staff.html")


@app.route("/ai-predictions")
def ai_predictions():
    return render_template("ai_predictions.html")


@app.route("/resource-transfer")
def resource_transfer():
    return render_template("resource_transfer.html")


@app.route("/settings")
def settings():
    return render_template("settings.html")


if __name__ == "__main__":
    app.run(debug=True)
