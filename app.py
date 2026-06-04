from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Tic Tac Toe</title>

<style>
body{
    font-family:Arial;
    text-align:center;
    margin-top:30px;
}

.board{
    width:306px;
    margin:auto;
    display:grid;
    grid-template-columns:repeat(3,100px);
    gap:3px;
}

.cell{
    width:100px;
    height:100px;
    font-size:50px;
    cursor:pointer;
}
</style>

</head>

<body>

<h1>Tic Tac Toe</h1>

<h3 id="status">Player X Turn</h3>

<div class="board">

<button class="cell"></button>
<button class="cell"></button>
<button class="cell"></button>

<button class="cell"></button>
<button class="cell"></button>
<button class="cell"></button>

<button class="cell"></button>
<button class="cell"></button>
<button class="cell"></button>

</div>

<br>

<button onclick="restartGame()">Restart</button>

<script>

let currentPlayer = "X";

const cells = document.querySelectorAll(".cell");

cells.forEach(cell => {

    cell.addEventListener("click", () => {

        if(cell.innerHTML === ""){

            cell.innerHTML = currentPlayer;

            currentPlayer =
                currentPlayer === "X" ? "O" : "X";

            document.getElementById("status").innerHTML =
                "Player " + currentPlayer + " Turn";
        }

    });

});

function restartGame(){

    cells.forEach(cell=>{
        cell.innerHTML="";
    });

    currentPlayer="X";

    document.getElementById("status").innerHTML =
        "Player X Turn";
}

</script>

</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)