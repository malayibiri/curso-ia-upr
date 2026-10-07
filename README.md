# curso-ia-upr

Sitio web del curso **Introducción a la Inteligencia Artificial**
(2026–2027 · Ingeniería Informática, Universidad de Pinar del Río
"Hermanos Saíz Montes de Oca").

Publicado en <https://malayibiri.github.io/curso-ia-upr/> mediante
GitHub Pages (workflow en `.github/workflows/deploy.yml`, despliegue
automático en cada push a `main`).

## Secciones del sitio

| Carpeta | Sección | Contenido |
|---|---|---|
| `01-contenido/` | Contenido | 5 temas con demostraciones Python ejecutables en el navegador |
| `02-tareas/` | Tareas | 5 trabajos prácticos del curso |
| `03-problemas/` | Problemas | Banco de 65 enunciados de ediciones pasadas (9 con vínculo local) |
| `04-proyecto/` | Proyecto de Curso | Orientaciones del trabajo final |
| `assets/` | — | Widget Pyodide (`pyodide.html`) y estilos |

## Comandos

```bash
quarto render            # compila todo el sitio a public/
quarto preview           # vista previa local con recarga
quarto render 03-problemas/  # solo una sección (rápido)
```

Requisito: Quarto ≥ 1.8. No hay dependencias Python
(`execute.eval: false`); no usar celdas ejecutables `{python}`.

## Convenciones (ver `AGENTS.md`)

- Bloques de código **siempre con lenguaje** (` ```text `, ` ```python `):
  una cerca sin lenguaje rompe `quarto render`.
- Fichas de `03-problemas/`: **anonimato total** (sin nombres, cursos ni años)
  y **enunciado, no solución**, con frontmatter `tipo-ia` + `local`.
- Al añadir páginas, registrarlas en el `sidebar` de `_quarto.yml`.
- No commitear `public/` (está en `.gitignore`).

## Recursos del curso

- [AIMA (Berkeley)](https://aima.cs.berkeley.edu/)
- [Understanding Deep Learning](https://github.com/udlbook)
- [MLU-Explain](https://mlu-explain.github.io/)
- [Dive into Deep Learning](https://d2l.ai/)
- [LLM visual](https://bbycroft.net/llm)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [CS231n (Stanford)](https://cs231n.stanford.edu/)
