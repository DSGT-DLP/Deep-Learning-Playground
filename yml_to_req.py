""" 
Utility file to convert our environment.yml file to a requirements.txt for deployment purpose. 

Credit to: https://gist.github.com/pemagrg1/f959c19ec18fee3ce2ff9b3b86b67c16

"""

import ruamel.yaml

yaml = ruamel.yaml.YAML()
data = yaml.load(open('conda/environment.yml'))

requirements = []
for dep in data['dependencies']:
    if isinstance(dep, str):
        if (dep == "pytorch"):
            requirements.append("torch")
        else:
            requirements.append(dep)
    elif isinstance(dep, dict):
        for preq in dep.get('pip', []):
            requirements.append(preq)

with open('requirements.txt', 'w') as fp:
    for requirement in requirements:
        fp.write(requirement + "\n")