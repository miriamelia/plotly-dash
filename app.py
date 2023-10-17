from dash import Dash

# App idea is to present a prototype of a monitoring UI for a ML model in the medical context

app = Dash(__name__)

# Design App here and add content from figures, dataset and other relevant calculations to respective divs


if __name__ == "__main__":
    app.run(debug=True)