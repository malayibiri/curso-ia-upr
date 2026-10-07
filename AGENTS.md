# AGENTS.md — curso-ia-upr

Sitio Quarto del curso *Introducción a la Inteligencia Artificial* (UPR).
Compila a `public/` y se despliega a GitHub Pages vía Actions en cada push a `main`.

## Comandos

```bash
quarto render            # compila todo el sitio a public/
quarto preview           # vista previa local con recarga
quarto render proyectos/ # solo la sección de proyectos (rápido)
```

Requisito: Quarto ≥ 1.8. No hay dependencias Python: `execute.eval: false`
en `_quarto.yml`; no usar celdas ejecutables `{python}`.

## Reglas de contenido

- **Bloques de código siempre con lenguaje** (p. ej. ` ```text `, ` ```python `).
  Una cerca sin lenguaje hace que Quarto intente arrancar un kernel y rompe
  `quarto render`. Los ` ```python ` son solo display (no se ejecutan).
- Todo el contenido va en `.qmd` con frontmatter `title` (+ `format: html` opcional).

## Sección `proyectos/` (banco de enunciados)

- Una ficha por proyecto: `proyectos/<año>-<tema>.qmd` + entrada en
  `proyectos/index.qmd` (tabla local y tabla de su tipo IA).
- **Anonimato total**: nunca nombres de estudiantes, cursos ni años en el texto.
- **Enunciado, no solución**: situación, objetivo, datos, entregables. Sin
  código, resultados ni conclusiones del trabajo original.
- Frontmatter obligatorio:

```yaml
---
title: "<problema, sin autor>"
tipo-ia: <clave>
local: true/false
---
```

- Claves `tipo-ia`: `búsqueda-no-informada`, `búsqueda-informada-A*`,
  `metaheurísticas`, `CSP/juegos`, `clasificación`, `regresión/predicción`,
  `recomendación`, `clustering`, `visión-CNN`, `NLP/LLM`, `RL/juegos`,
  `agente-lógico/experto`, `serie-temporal/optimización`.
- `local: true` **solo** con mención explícita a Pinar del Río/Cuba; citar la
  frase como `> Vínculo local: "..."` al inicio del cuerpo.
- Plantilla de secciones: Situación problemática, Objetivo, Datos y entorno,
  Técnicas de IA sugeridas, Alcance y entregables, Métricas sugeridas, Extensiones.
- Al añadir una ficha, registrarla también en el `sidebar` de `_quarto.yml`
  dentro de su sección por tipo IA.

## Widget Pyodide

- Fuente única: `assets/pyodide.html`, inyectado globalmente vía
  `format.html.include-after-body`. No añadir `<script>` manuales en páginas.
- El include lleva guarda: solo descarga Pyodide (CDN jsDelivr) en páginas con
  `.pyodide-cell`. Estructura de celda: `div.pyodide-widget` con
  `button.run` + `textarea` + `pre.output`.
- Limitaciones conocidas del widget: sin `input()` interactivo, sin
  `matplotlib.show()`; documentarlas en la propia página si se proponen
  ejercicios que las usen.

## Despliegue

Workflow `.github/workflows/deploy.yml`: checkout → setup Quarto →
`quarto render` → artifact `public` → Pages. No commitear `public/`
(está en `.gitignore`).
