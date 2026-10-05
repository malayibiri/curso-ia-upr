# curso-ia-upr

Sitio web del curso de **Introducción a la Inteligencia Artificial** de la
Universidad de Pinar del Río "Hermanos Saíz Montes de OCa".

## Recursos

- https://aima.cs.berkeley.edu/
- https://github.com/udlbook
- https://mlu-explain.github.io/
- https://d2l.ai/
- https://bbycroft.net/llm
- https://poloclub.github.io/transformer-explainer/
- https://poloclub.github.io/
- https://cs231n.stanford.edu/

## Estructura

```
.quarto.yml          # config del sitio
index.qmd            # portada
01-introduccion/     # presentación del curso
02-aprendizaje/      # capítulos temáticos
  01/                # búsqueda
  02/                # panorama de la IA
  03/                # demo Pyodide
03-proyectos/        # proyectos integradores
tareas/              # 5 tareas prácticas
assets/              # JS y CSS
styles.css           # estilos del sitio
.github/workflows/   # despliegue a GitHub Pages
```

## Compilar localmente

```bash
quarto render
# el sitio queda en public/
```

## Despliegue

El sitio se despliega automáticamente a
<https://malayibiri.github.io/curso-ia-upr/> mediante GitHub Actions
(workflow en `.github/workflows/deploy.yml`) en cada push a `main`.

> Para habilitar Pages: en el repo GitHub → Settings → Pages → Deploy from
> branch → main → `public` (artifact).
