// Site-level JS — carga el widget de Pyodide en todas las páginas.
// Inyectado por Quarto vía _quarto.yml (website: css/scripts).
// El widget en sí (assets/pyodide-widget.js) se carga por este script.
document.addEventListener("DOMContentLoaded", function () {
  // Si la página tiene celdas Pyodide, cargar el widget
  if (document.querySelectorAll(".pyodide-cell").length > 0) {
    const s = document.createElement("script");
    s.src =
      (document.baseURI.replace(/[^/]*$/, "")) + "assets/pyodide-widget.js";
    document.body.appendChild(s);
  }
});
