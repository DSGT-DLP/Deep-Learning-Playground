import os;

directories = ['frontend', 'backend']

ignored = ['.next', 'node_modules', '.venv']


def traverse_directory(dir: str, is_root: bool, prefix: str) -> str:
    print("Traversing subdir: " + dir)
    os.chdir(dir)
    
    output = ""
    if is_root:
        output = '* ' + dir
    else:
        output = prefix + dir + '/'
    output += '\n'
    
    next_prefix = '|  ' + prefix
    
    subdirs = os.listdir()
    for subdir in subdirs:
        if os.path.isfile(subdir):
            output += prefix + subdir
            output += '\n'
        elif subdir not in ignored:
            subtree = traverse_directory(subdir, False, next_prefix)
            output += subtree
            
    os.chdir('..')
    return output


output = "#Architecture\n\n"
for directory in directories:
    output += "## " + directory.capitalize() + " Architecture\n\n"
    output += "```\n"
    output += traverse_directory(directory, True, "|- ")
    output += "```"
    output += "\n\n"
print(output)
#write output to file
