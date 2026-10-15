# Authoring the course

The `../curriculum/lab/` modules are the portable learner copies. These files
include the authoring source and generator. From the course directory, install
the core requirements plus `nbformat` and `markdown-it-py`, then run:

```bash
python lab/build_curriculum.py
```

This regenerates the reader content, notes and unexecuted notebooks. To regenerate
outputs, first fetch model assets with `python lab/setup_robot_assets.py`, then
use `python lab/build_curriculum.py --execute` in an appropriate runtime. Update
the verification reports only after executing the corresponding checks.
