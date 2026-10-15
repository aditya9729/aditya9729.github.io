# Publish Embodied RL on GitHub Pages

The course is prepared under `courses/embodied-rl/` in the public review branch
`embodied-rl-course-2026` of `aditya9729/aditya9729.github.io`.
The existing portfolio homepage is not replaced. The proposed course path after
merging is `https://aditya9729.github.io/courses/embodied-rl/`.
That is a proposed Pages URL until the branch is merged and the repository's
Pages build succeeds; the private preview is the current deployed review copy.

## Review checklist

1. Review the pull request's course text, author attribution, visual hierarchy,
   model controls, research labels and installation evidence.
2. Check the notebook links. They use the public review branch and open actual
   `.ipynb` files in Colab now; each notebook embeds its editable modules.
3. Merge the PR when ready. The existing repository's Pages mechanism must deploy
   the root directory or copy the course folder into its build output. Existing
   framework filtering may require an explicit static-copy step.
4. After a successful Pages build, verify the proposed course URL on desktop and
   mobile, keyboard navigation, equation overflow, notebook opening, and downloads.
5. Keep the source branch or switch `course-config.js` to `ref:'main'` after merge.
   Colab links have one configuration object; they do not depend on the Site host.

All reader assets use relative paths, so the `/courses/embodied-rl/` subpath works.
KaTeX and its fonts are local, with its upstream MIT license retained. Google Fonts
have system-font fallbacks. Progress and notes remain browser-local and do not sync
between the private preview and GitHub Pages origins. Export notes before switching.

The GitHub copy downloads the source branch ZIP instead of depending on the private
preview's learning ZIP. After extracting that repository ZIP, enter
`courses/embodied-rl/curriculum/` to run the installers and notebooks.
The native Site copy offers the convenient prepackaged learning ZIP instead.

## Current verification boundary

Logic/math, actual Linux CPU notebook execution, native MuJoCo integration, standalone
module restoration, and short SB3 baseline execution are checked. No supported browser
controller was available, so this release does not claim visual browser QA. Mac and
Colab CUDA require platform execution. Robot source assets download separately and
retain their original licenses. The optional R1 base is idealized planar motion.

## Editorial references

Original explanatory text and interactive diagrams use question → derivation →
numeric case → update trace → experiment. Stanford CS224R and Berkeley CS185/285
provide useful external lecture progressions and complementary advanced study.
Shengyu Feng's Beyond RL motivates the optional sampling/optimization comparison.
These links are references, not affiliation or endorsement:

- https://cs224r.stanford.edu/
- https://rail.eecs.berkeley.edu/deeprlcourse/
- https://shengyu-feng.github.io/blog/2026/08/27/beyond-RL/
