// ============================================================
//  Pyodide widget — celdas interactivas de Python en el navegador
//  Carga Pyodide desde el CDN de jsDelivr.
//  Uso:
//    <div class="pyodide-widget">
//      <button class="run">▶ Ejecutar</button>
//      <textarea>print("hola")</textarea>
//      <pre class="output"></pre>
//    </div>
// ============================================================
(async function () {
  if (window.__pyodideLoaded) return;

  // Cargar Pyodide desde CDN
  const script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
  script.onload = initAll;
  document.head.appendChild(script);

  function initAll() {
    window.__pyodideLoaded = true;
    loadPyodide().then((pyodide) => {
      window.__pyodide = pyodide;
      document.querySelectorAll(".pyodide-widget").forEach(bindCell);
    });
  }

  function bindCell(widget) {
    const ta = widget.querySelector("textarea");
    const out = widget.querySelector(".output");
    const btn = widget.querySelector("button.run");

    btn.addEventListener("click", async () => {
      out.textContent = "";
      const code = ta.value;
      try {
        // Redirigir stdout a una variable que podamos leer
        await window.__pyodide.runPythonAsync(`
import sys, io
_capture = io.StringIO()
_sys_stdout = sys.stdout
sys.stdout = _capture
try:
    exec(${JSON.stringify(code)})
except Exception as e:
    _capture.write(f"{type(e).__name__}: {e}")
finally:
    sys.stdout = _sys_stdout
# Devolver el contenido capturado a JS
get_capture_result = lambda: _capture.getvalue()
        `);
        // Leer el resultado capturado
        const result = await window.__pyodide.runPythonAsync(
          "get_capture_result()"
        );
        out.textContent = result;
      } catch (err) {
        out.textContent = "Error: " + err;
      }
    });
  }
})();