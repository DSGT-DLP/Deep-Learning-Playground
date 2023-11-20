export const isNameValid = (name: string) : boolean => {
    name = name.toLowerCase();
    for (let i = 0; i < name.length; ++i) {
        const c = name.charAt(i);
        /* Checks if a character is a letter [a-z] or [A - Z] */
        if (c.toLowerCase() === c.toUpperCase()) {
            return false;
        }
    }
    return true;
};
