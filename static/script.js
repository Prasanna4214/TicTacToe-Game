const cells = document.querySelectorAll(".cell");

let currentPlayer = "X";

cells.forEach(cell => {
    cell.addEventListener("click", () => {
        if(cell.textContent === ""){
            cell.textContent = currentPlayer;
            currentPlayer = currentPlayer === "X" ? "O" : "X";
        }
    });
});