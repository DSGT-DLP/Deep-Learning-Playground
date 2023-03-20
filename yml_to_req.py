""" 
Utility file to convert our environment.yml file to a requirements.txt for deployment purpose. 

Credit to: https://gist.github.com/pemagrg1/f959c19ec18fee3ce2ff9b3b86b67c16

"""

import ruamel.yaml

yaml = ruamel.yaml.YAML()
data = yaml.load(open("conda/environment.yml"))

requirements = []
for dep in data["dependencies"]:
    if isinstance(dep, str):
        if dep == "pytorch":
            requirements.append("torch")
        elif dep.startswith("python"):
            continue
        else:
            parsed_dep = dep.split("=")
            if parsed_dep[0] == dep:
                requirements.append(dep)
            elif len(parsed_dep) > 2:
                raise ValueError(
                    "Misformat in environment.yml file. Make sure that you have <packageName>=<version> or <packageName> format"
                )
            else:
                package_name, package_version = parsed_dep[0], parsed_dep[1]
                requirements.append(f"{parsed_dep[0]}=={parsed_dep[1]}")
    elif isinstance(dep, dict):
        for preq in dep.get("pip", []):
            requirements.append(preq)

with open("requirements.txt", "w") as fp:
    for requirement in requirements:
        fp.write(requirement + "\n")
