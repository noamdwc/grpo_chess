### 0. Intro
planning with me will take around 10 questions. i know it’s a bit of a stretch, but trust me... it’s worth it. the better your answers are, the better your app will be.

we can either take 5 minutes now to plan it right and get a working app with almost no edits…
or we can rush it, and you’ll end up with something that doesn’t quite work unless you’re a pro.

totally your call... i’ll guide you every step of the way.

### 1. User expertise
you're an experienced data scientist with 5 years in time series classification, including 2 years as a ds team lead.

### 2. Distilling DeepMind's Searchless Chess
you're embarking on a new project to distill a model from deepmind's searchless chess project, located at `https://github.com/google-deepmind/searchless_chess`.

### Causing 1 issue
since this is a fun side project, the main implication of not solving this is missing out on the enjoyment and learning experience of diving into the deepmind project.

### Compare Distillation Options
you want to be able to compare two or three different distillation options for the deepmind searchless chess model.

### The benefits include:

- adding this project to your cv.
- gaining deeper insights into deep learning and distillation techniques.

### 6. Required Capabilities

- the system must be able to download the deepmind project's models and data.
- you need to build your own smaller model.
- the solution should distill deepmind's model into your smaller model.
- you'll be leveraging an existing, smaller chess bot model you already have.

### 7. Recommended Tech Stack

- **pytorch** — for building and training your deep learning models, offering flexibility and control.
- **wandb** — for experiment tracking, visualization, and comparing your different distillation runs.
- **pytorch lightning** — to structure your training code, making it easier to scale from a single gpu to multi-gpu setups and simplifying boilerplate.
- **lightning studio** — runs and hosts everything. no setup required, provides a persistent environment, and has gpu support.

### 8. Project Milestones and Work Plan

# Phase 0 - MVP
Purpose: fully unblock you to start experimenting with the deepmind project and your own model.

- [x] download a small model from deepmind.
- [x] download a sample of the dataset.
- [x] train a simple small model of your own.

# Phase 1 - polished product
Purpose: improve model performance and data coverage.

- [x] train on the entire dataset.

# Phase 2+ - scale or extend
Purpose: address new use cases or explore advanced distillation techniques.

- [x] add new methods of distillation.

### 10. How to deploy it
since this is a research project, it will be run once to generate and save the results, plots, and new models.