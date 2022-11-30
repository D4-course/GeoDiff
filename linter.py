from pylint.lint import Run
pylint_opts = ['--version']
import os
for file in os.listdir("models/epsnet"):
    if file.endswith(".py"):
        results = Run([os.path.join("models/epsnet", file), "--rcfile=./.pylintrc"], do_exit=False)
        # exit()

for file in os.listdir("utils"):
    if file.endswith(".py"):
        results = Run([os.path.join("utils", file), "--rcfile=./.pylintrc"], do_exit=False)

# for file in os.listdir("."):
#     if file.endswith(".py"):
#         results = Run([os.path.join(".", file), "--rcfile=./.pylintrc"], do_exit=False)

print(results.linter.stats['global_note'])